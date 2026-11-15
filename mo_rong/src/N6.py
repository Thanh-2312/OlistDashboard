# %% [markdown]
# # N6 – LAKEHOUSE MỞ RỘNG: DỮ LIỆU BÁN CẤU TRÚC (JSON, API) VÀ KHÔNG CẤU TRÚC (VĂN BẢN)
#
# **Vì sao cần notebook này?** Kho SQLite (N3) và vùng Gold (N5) mới chứa **dữ liệu có cấu trúc** (bảng). Điểm mạnh thật sự của
# Data Lakehouse so với kho dữ liệu là **một hồ chứa được cả ba loại dữ liệu** mà vẫn giữ giao dịch ACID, phiên bản, kiểm soát lược đồ
# (giáo trình 2.4.2; Bài 1.1 – bảng so sánh *Data warehouse / Data lake / Data lakehouse*). Notebook này chứng minh điều đó trên dữ liệu thật:
#
# | Loại dữ liệu | Nguồn trong bài | Vấn đề nếu đưa thẳng vào kho truyền thống | Cách lakehouse giải quyết (notebook này) |
# |---|---|---|---|
# | Bán cấu trúc – API | 3 API công khai của Brazil: tỷ giá USD/BRL (Ngân hàng Trung ương), dân số bang và thành phố (IBGE), ngày lễ (BrasilAPI) | Mỗi API một dạng JSON: số và ngày lưu dạng chữ, dòng tiêu đề lẫn trong dữ liệu, giá trị thiếu ghi `...`, tên ghép `"Thành phố - Bang"` | Vùng hạ cánh giữ file gốc; Bronze lưu nguyên văn, nạp lại không trùng (MERGE); hỏi được ngay bằng SQL (schema-on-read); Silver mới ép kiểu |
# | Bán cấu trúc – JSON lồng nhau | 99.441 đơn hàng dạng tài liệu JSON (như webhook của sàn): khách hàng lồng địa chỉ, mảng sản phẩm, mảng thanh toán | Lồng nhiều cấp, mảng; **API đổi phiên bản** (thêm trường, dời trường, tiền gửi dạng chữ `"58,90"`); dòng hỏng | Delta lưu cột struct / list; **schema evolution** tự thêm trường; **schema enforcement** + ACID chặn sai kiểu mà không ghi dở; cách ly dòng lỗi; DuckDB `UNNEST` làm phẳng |
# | Không cấu trúc – văn bản | Hơn 42 nghìn bình luận đánh giá tiếng Bồ Đào Nha | Chữ tự do, xuống dòng, emoji, số điện thoại của khách (dữ liệu cá nhân), câu trùng lặp | Silver chuẩn hóa và **che dữ liệu cá nhân**; Gold tách từ thành **chỉ mục tìm kiếm** và nối với bảng sự kiện; N7 dùng AI đọc nội dung |
#
# **Chạy sau N5** (N5 dựng lại toàn bộ hồ dữ liệu). File gốc nhận từ API nằm ở `data/landing/` (ngoài hồ dữ liệu) nên chạy lại không cần mạng.
#
# | Phần | Job | Nội dung |
# |---|---|---|
# | 1 | – | Chuẩn bị |
# | 2 | L13 → L16 | API → Landing → Bronze → Silver → Gold: tỷ giá, dân số, ngày lễ |
# | 3 | L17 → L20 | Đơn hàng dạng JSON lồng nhau: hai phiên bản API, thay đổi lược đồ, dòng lỗi, làm phẳng, đối soát, so tốc độ |
# | 4 | L21 → L22 | Văn bản đánh giá: làm sạch, che dữ liệu cá nhân, chỉ mục từ, nối với dữ liệu giao hàng |
# | 5 | – | Tổng kết: vấn đề → giải pháp → bằng chứng |

# %%
import time, json, hashlib, re, shutil, unicodedata, warnings
from contextlib import contextmanager
from datetime import datetime
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import duckdb, requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from deltalake import DeltaTable, write_deltalake
from IPython.display import display
from olist_utils import *

warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 90)

LAKE = DATA / 'lakehouse'
LANDING = DATA / 'landing'                        # vùng hạ cánh: file gốc nhận từ API / hệ thống nguồn (N5 không xóa thư mục này)
CHART = PROJECT / 'assets' / 'charts'
CHART.mkdir(parents=True, exist_ok=True)
if not (LAKE / 'gold' / 'fact_order').exists():
    raise RuntimeError('Chưa có vùng Gold – hãy chạy N5_Lakehouse trước')
REFRESH_API = False                               # True = gọi lại API kể cả khi đã có file trong vùng hạ cánh
N6_TABLES = ['bronze/api_raw', 'bronze/order_events_raw', 'silver/fx_usd_brl', 'silver/population_uf', 'silver/population_city',
             'silver/holiday_api', 'silver/order_events', 'silver/order_events_quarantine', 'silver/order_events_items',
             'silver/order_events_payments', 'silver/review_text', 'gold/dm_sales_monthly_usd', 'gold/dm_state_market',
             'gold/fact_review_text', 'gold/review_token_index']
for t in N6_TABLES:                               # chạy lại = dựng lại các bảng của N6, không đụng bảng của N5
    shutil.rmtree(LAKE / t, ignore_errors=True)
BATCH = 'B' + pd.Timestamp.now().strftime('%Y%m%d_%H%M')
NOW = lambda: pd.Timestamp.now().isoformat(timespec='seconds')
BLUE, ORANGE, GREEN, RED, GREY = '#1F5FA8', '#E07B1A', '#2E9E5B', '#C0392B', '#8C8C8C'
LOG = []


@contextmanager
def job(jid, name, layer, inp, out):
    """Một job của pipeline: đo thời gian, đếm dòng vào / ra (cùng quy ước với N5)."""
    info = dict(rows_in=0, rows_out=0)
    t0 = time.perf_counter()
    yield info
    sec = round(time.perf_counter() - t0, 2)
    LOG.append(dict(job_id=jid, job_name=name, layer=layer, input=inp, output=out, rows_in=int(info['rows_in']), rows_out=int(info['rows_out']), seconds=sec))
    print(f'{jid} · {name}: {int(info["rows_in"]):,} → {int(info["rows_out"]):,} dòng · {sec}s')


def to_arrow(df):
    """DataFrame -> bảng Arrow đúng kiểu Delta: thời gian về micro-giây, cột toàn rỗng thành chuỗi."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype('datetime64[us]')
    t = pa.Table.from_pandas(df, preserve_index=False)
    for i, f in enumerate(t.schema):
        if pa.types.is_null(f.type):
            t = t.set_column(i, f.name, t.column(i).cast(pa.string()))
    return t


def put(df, layer, name, mode='overwrite'):
    """Ghi một bảng Delta: overwrite = thay toàn bộ, append = thêm một phiên bản (lược đồ được phép mở rộng)."""
    write_deltalake(str(LAKE / layer / name), to_arrow(df), mode=mode, schema_mode='overwrite' if mode == 'overwrite' else 'merge')
    return len(df)


def get(layer, name, **kw):
    return DeltaTable(str(LAKE / layer / name), **kw).to_pandas()


def dset(layer, name):
    """Bảng Delta dưới dạng Arrow dataset – DuckDB truy vấn trực tiếp bằng tên biến."""
    return DeltaTable(str(LAKE / layer / name)).to_pyarrow_dataset()


def folder_mb(p):
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6


def bench(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, round(float(np.median(ts)) * 1000, 1)


def leaf_fields(t, prefix=''):
    """Liệt kê mọi trường lá của một lược đồ lồng nhau: 'customer.address.state', 'items[].price'..."""
    out = []
    for f in t:
        name, ty = prefix + f.name, f.type
        if pa.types.is_list(ty):
            name, ty = name + '[]', ty.value_type
        out += leaf_fields(ty, name + '.') if pa.types.is_struct(ty) else [name]
    return out


print('Hồ dữ liệu:', LAKE.relative_to(PROJECT).as_posix(), '· vùng hạ cánh:', LANDING.relative_to(PROJECT).as_posix(), '· batch', BATCH)

# %% [markdown]
# ## 2. API → LANDING → BRONZE → SILVER → GOLD
# ### L13 – Thu thập dữ liệu thật qua API (Extract)
# | Nguồn | Dữ liệu | Dùng để |
# |---|---|---|
# | Ngân hàng Trung ương Brazil – API SGS, chuỗi số 1 | Tỷ giá USD/BRL hằng ngày 09/2016 → 10/2018 | Quy đổi doanh thu sang USD đúng tỷ giá ngày đặt hàng |
# | IBGE – API SIDRA, bảng 6579 | Dân số ước tính của 27 bang (2017, 2018) và 5.570 thành phố (2018) | Số đơn / 100.000 dân; kiểm tra tên thành phố trong dữ liệu Olist |
# | BrasilAPI | Ngày lễ quốc gia 2016, 2017, 2018 | Đối chiếu cột ngày lễ của `dim_date` |
#
# Mỗi phản hồi được lưu **nguyên văn** thành một file trong `data/landing/api/` trước khi xử lý (*landing zone*).
# Đã có file thì dùng lại (không gọi mạng); API lỗi mà chưa có file thì dừng và báo rõ.

# %%
API = [('bcb', 'usd_brl_daily', 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=json&dataInicial=01/09/2016&dataFinal=31/10/2018',
        'Tỷ giá USD/BRL hằng ngày (BCB SGS 1)'),
       ('ibge_sidra', 'population_uf', 'https://apisidra.ibge.gov.br/values/t/6579/n3/all/v/9324/p/2017,2018', 'Dân số ước tính 27 bang, 2017–2018'),
       ('ibge_sidra', 'population_city', 'https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2018', 'Dân số ước tính 5.570 thành phố, 2018')] + \
      [('brasilapi', f'holidays_{y}', f'https://brasilapi.com.br/api/feriados/v1/{y}', f'Ngày lễ quốc gia {y}') for y in (2016, 2017, 2018)]


def fetch(src, name, url):
    """Gọi API một lần, lưu nguyên văn vào landing; đã có file thì dùng lại."""
    f = LANDING / 'api' / src / f'{name}.json'
    if f.exists() and not REFRESH_API:
        return f, 'đọc lại file đã lưu'
    try:
        r = requests.get(url, headers={'User-Agent': 'olist-dw-student-project/1.0'}, timeout=(20, 120))
        r.raise_for_status()
        json.loads(r.content)                     # chỉ lưu khi đúng là JSON
    except Exception as e:
        if f.exists():
            return f, f'API lỗi ({type(e).__name__}) → dùng file đã lưu'
        raise RuntimeError(f'Không gọi được {url} và chưa có file trong {f.parent}') from e
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(r.content)
    return f, f'gọi API ({r.elapsed.total_seconds():.1f}s)'


with job('L13', 'API → Landing (file JSON nguyên văn)', 'Landing', f'{len(API)} lời gọi API', 'data/landing/api/*.json') as J:
    MANI = []
    for src, name, url, desc in API:
        f, how = fetch(src, name, url)
        body = f.read_bytes()
        d = json.loads(body)
        MANI.append(dict(source=src, dataset=name, description=desc, file=f.relative_to(DATA).as_posix(), kb=round(len(body) / 1024, 1),
                         records=len(d) if isinstance(d, list) else 1, sha256=hashlib.sha256(body).hexdigest()[:16],
                         saved_at=pd.Timestamp.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M'), mode=how, url=url))
        J['rows_in'] += 1; J['rows_out'] += MANI[-1]['records']
mani = save(pd.DataFrame(MANI), 'lx_api_manifest')
display(mani.drop(columns=['url', 'file']))

# %%
# Ba API, ba "hình dạng" JSON khác nhau – đặc trưng của dữ liệu bán cấu trúc (in 2 phần tử đầu của mỗi phản hồi)
for src, name, *_ in API[:4]:
    d = json.loads((LANDING / 'api' / src / f'{name}.json').read_text(encoding='utf-8'))
    print(f'{name}:', json.dumps(d[:2], ensure_ascii=False)[:330], '\n')

# %% [markdown]
# ### L14 – Bronze: lưu nguyên văn từng bản ghi JSON, nạp lại không bị trùng
# Mỗi phần tử của mảng JSON thành một dòng; cột `raw_json` giữ **nguyên văn** (chưa ép kiểu, chưa đổi tên) cùng cột truy vết (nguồn, file, mã băm).
# Nạp bằng **MERGE theo (mã băm file, số thứ tự bản ghi)**: chạy lại với cùng file thì không thêm dòng nào – nạp **lũy đẳng (idempotent)**.
# Kho SQLite chỉ làm được việc này nếu tự viết thêm logic kiểm tra trùng.

# %%
API_RAW = str(LAKE / 'bronze' / 'api_raw')


def ingest_api(m):
    d = json.loads((DATA / m['file']).read_text(encoding='utf-8'))
    df = pd.DataFrame(dict(source=m['source'], dataset=m['dataset'], file=m['file'], file_sha256=m['sha256'], record_no=np.arange(len(d)),
                           raw_json=[json.dumps(x, ensure_ascii=False) for x in d], _batch_id=BATCH, _load_ts=NOW()))
    if not DeltaTable.is_deltatable(API_RAW):
        put(df, 'bronze', 'api_raw')
        return len(df)
    r = DeltaTable(API_RAW).merge(to_arrow(df), predicate='t.file_sha256 = s.file_sha256 AND t.record_no = s.record_no',
                                  source_alias='s', target_alias='t').when_not_matched_insert_all().execute()
    return r['num_target_rows_inserted']


with job('L14', 'Landing → Bronze api_raw (MERGE, lũy đẳng)', 'Bronze', 'data/landing/api/*.json', 'bronze/api_raw') as J:
    first = [ingest_api(m) for m in MANI]
    again = [ingest_api(m) for m in MANI]         # nạp lại đúng các file đó một lần nữa
    J['rows_in'] = sum(first) + sum(again)
    J['rows_out'] = dset('bronze', 'api_raw').count_rows()
idem = save(pd.DataFrame(dict(dataset=[m['dataset'] for m in MANI], records=[m['records'] for m in MANI], lan_1_them=first, lan_2_them=again)), 'lx_idempotent')
display(idem)
assert sum(again) == 0 and sum(first) == J['rows_out'], 'nạp lại bị trùng dòng'
print('Lịch sử bảng bronze/api_raw:', pd.DataFrame(DeltaTable(API_RAW).history()).operation.value_counts().to_dict())

# %%
# Schema-on-read: hỏi thẳng trên chuỗi JSON thô của Bronze bằng SQL, chưa cần tạo bảng có cấu trúc
api_raw = dset('bronze', 'api_raw')
display(duckdb.sql("""
    SELECT right(json_extract_string(raw_json, '$.data'), 4) AS nam,
           ROUND(AVG(CAST(json_extract_string(raw_json, '$.valor') AS DOUBLE)), 4) AS ty_gia_usd_brl_tb,
           COUNT(*) AS so_ngay_co_ty_gia
    FROM api_raw WHERE dataset = 'usd_brl_daily' GROUP BY 1 ORDER BY 1""").df())

# %% [markdown]
# ### L15 – Silver: ép kiểu và xử lý các "bẫy" thật của từng API
# | API | Bẫy dữ liệu gặp thật | Cách xử lý |
# |---|---|---|
# | Tỷ giá BCB | Ngày dạng chữ `dd/mm/yyyy`, tỷ giá dạng chữ; ngày nghỉ không có số liệu | Ép kiểu; dựng lịch đủ ngày, ngày nghỉ lấy tỷ giá ngày làm việc gần nhất trước đó (cờ `is_filled`) |
# | IBGE SIDRA | Phần tử đầu mảng là **dòng tiêu đề**, không phải dữ liệu; mã và số dạng chữ; dân số thiếu ghi `...`; tên thành phố ghép `"Tên - UF"` | Bỏ dòng tiêu đề theo nội dung; `...` → rỗng; tách tên và bang |
# | BrasilAPI | Mỗi năm một file, tên lễ tiếng Bồ Đào Nha | Gộp 3 năm, đối chiếu với `dim_date` |

# %%
raw = api_raw.to_table(columns=['dataset', 'record_no', 'raw_json']).to_pandas()
JS = lambda ds: pd.DataFrame([json.loads(x) for x in raw[raw.dataset == ds].sort_values('record_no').raw_json])
IBGE_REGION = {1: 'North', 2: 'Northeast', 3: 'Southeast', 4: 'South', 5: 'Central-West'}    # chữ số đầu của mã bang IBGE
NAME2UF = dict(zip(norm_text(pd.Series(list(STATE_NAME.values()))), STATE_NAME.keys()))


def sidra(ds):
    """Đọc phản hồi SIDRA: bỏ dòng tiêu đề, '...' (không có số liệu) thành rỗng."""
    d = JS(ds)
    head = d.V.eq('Valor')
    d = d[~head].copy()
    d['population'] = pd.to_numeric(d.V.where(~d.V.isin(['...', '..', '-', 'X'])), errors='raise').astype('Int64')
    return d, int(head.sum()), int(d.population.isna().sum())


with job('L15', 'Bronze → Silver: tỷ giá, dân số, ngày lễ', 'Silver', 'bronze/api_raw', 'silver/fx_usd_brl, population_uf, population_city, holiday_api') as J:
    d = JS('usd_brl_daily')
    fx = pd.DataFrame(dict(date=pd.to_datetime(d.data, format='%d/%m/%Y'), usd_brl=pd.to_numeric(d.valor)))
    fx = pd.DataFrame(dict(date=pd.date_range(fx.date.min(), fx.date.max(), freq='D'))).merge(fx, on='date', how='left')
    fx['is_filled'] = fx.usd_brl.isna().astype(int)
    fx['usd_brl'] = fx.usd_brl.ffill()
    fx['date_key'] = fx.date.dt.strftime('%Y%m%d').astype(int)

    d, head_uf, miss_uf = sidra('population_uf')
    pop_uf = pd.DataFrame(dict(uf_code=d.D1C.astype(int), state_name=d.D1N, state=norm_text(d.D1N).map(NAME2UF), year=d.D3N.astype(int),
                               population=d.population))
    pop_uf['region_ibge'] = (pop_uf.uf_code // 10).map(IBGE_REGION)
    assert pop_uf.state.notna().all(), 'có tên bang không khớp'

    d, head_city, miss_city = sidra('population_city')
    nm = d.D1N.str.rsplit(' - ', n=1, expand=True)
    pop_city = pd.DataFrame(dict(city_code=d.D1C.astype(int), city_name=nm[0], state=nm[1], population=d.population))
    pop_city['city_key'] = norm_text(pop_city.city_name) + '|' + pop_city.state

    d = pd.concat([JS(f'holidays_{y}') for y in (2016, 2017, 2018)], ignore_index=True)
    hol = pd.DataFrame(dict(date=pd.to_datetime(d['date']), holiday_pt=d['name'], type=d['type']))
    for name, df in [('fx_usd_brl', fx), ('population_uf', pop_uf), ('population_city', pop_city), ('holiday_api', hol)]:
        J['rows_out'] += put(df, 'silver', name)
    J['rows_in'] = len(raw)

quirks = save(pd.DataFrame([
    dict(api='BCB – tỷ giá', van_de='Ngày, tỷ giá lưu dạng chữ (dd/mm/yyyy, "3.2472")', so_luong=int((1 - fx.is_filled).sum()), xu_ly='Ép kiểu ngày và số'),
    dict(api='BCB – tỷ giá', van_de='Cuối tuần / ngày lễ không có tỷ giá', so_luong=int(fx.is_filled.sum()), xu_ly='Lấy tỷ giá ngày làm việc trước đó, gắn cờ is_filled'),
    dict(api='IBGE SIDRA', van_de='Dòng tiêu đề nằm trong mảng dữ liệu', so_luong=head_uf + head_city, xu_ly='Nhận ra theo nội dung (V = "Valor") rồi bỏ'),
    dict(api='IBGE SIDRA', van_de='Dân số thiếu, ghi "..."', so_luong=miss_uf + miss_city, xu_ly='Để rỗng, không tính vào mẫu số'),
    dict(api='IBGE SIDRA', van_de='Tên ghép "Thành phố - UF"', so_luong=len(pop_city), xu_ly='Tách thành city_name + state, tạo khóa so khớp không dấu'),
    dict(api='IBGE SIDRA', van_de='Bang xếp vùng khác với bảng vùng của nhóm', so_luong=int((pop_uf.region_ibge != pop_uf.state.map(REGION)).sum()),
         xu_ly='Đối chiếu theo chữ số đầu mã bang IBGE')]), 'lx_api_quirks')
display(quirks)

# %%
# Đối chiếu ngày lễ: BrasilAPI và cột is_holiday của dim_date (N5 tự tính bằng quy tắc Phục sinh)
dd = get('gold', 'dim_date')
h_dim = dd[dd.is_holiday == 1][['full_date', 'holiday_name']].assign(date=lambda x: pd.to_datetime(x.full_date)).drop(columns='full_date')
hc = h_dim.merge(hol, on='date', how='outer', indicator=True)
hc['ket_qua'] = hc._merge.map({'both': 'Khớp', 'left_only': 'Chỉ có trong dim_date', 'right_only': 'Chỉ có trong API'})
hc = save(hc.drop(columns='_merge').sort_values('date'), 'lx_holiday_check')
display(hc.ket_qua.value_counts().rename('so_ngay').to_frame())
display(hc[hc.ket_qua != 'Khớp'])

# %% [markdown]
# ### L16 – Gold: nối dữ liệu API với kho dữ liệu
# * `dm_sales_monthly_usd`: doanh thu từng tháng quy đổi USD **theo tỷ giá đúng ngày đặt hàng** (không dùng một tỷ giá trung bình).
# * `dm_state_market`: số khách, số đơn, doanh thu trên **100.000 dân** của 27 bang – độ phủ thị trường của Olist.
# * Danh mục 5.570 thành phố của IBGE dùng làm **chuẩn** để kiểm tra tên thành phố khách hàng / người bán trong dữ liệu Olist.

# %%
with job('L16', 'Silver + Gold → Gold: doanh thu USD, độ phủ thị trường', 'Gold', 'silver/fx_usd_brl, population_*; gold/fact_sales, dim_customer',
         'gold/dm_sales_monthly_usd, dm_state_market') as J:
    fs = get('gold', 'fact_sales')
    dc = get('gold', 'dim_customer')
    v = fs[fs.is_valid_sale == 1].merge(fx[['date_key', 'usd_brl']], on='date_key', how='left')
    assert v.usd_brl.notna().all(), 'có ngày bán không có tỷ giá'
    v['price_usd'] = v.price / v.usd_brl
    mon = v.groupby('year_month').agg(revenue_brl=('price', 'sum'), revenue_usd=('price_usd', 'sum'), orders=('order_id', 'nunique')).reset_index()
    mon['usd_brl_weighted'] = mon.revenue_brl / mon.revenue_usd           # tỷ giá bình quân theo doanh thu của tháng
    J['rows_out'] += put(mon, 'gold', 'dm_sales_monthly_usd')

    w = v[v.year_month.between('2017-01', '2018-08')].merge(dc[['customer_key', 'state']], on='customer_key')
    st = w.groupby('state').agg(customers=('customer_key', 'nunique'), orders=('order_id', 'nunique'), revenue_brl=('price', 'sum'),
                                revenue_usd=('price_usd', 'sum')).reset_index()
    st = st.merge(pop_uf[pop_uf.year == 2018][['state', 'state_name', 'population', 'region_ibge']], on='state', how='right').fillna(0)
    st['region_vn'] = st.region_ibge.map(REGION_VN)
    st['orders_per_100k'] = st.orders / st.population.astype(float) * 1e5
    st['customers_per_100k'] = st.customers / st.population.astype(float) * 1e5
    st['revenue_per_1000_brl'] = st.revenue_brl / st.population.astype(float) * 1000
    nat = st.orders.sum() / st.population.astype(float).sum() * 1e5
    st['index_vs_brazil'] = st.orders_per_100k / nat * 100                 # 100 = bằng mức trung bình cả nước
    st = st.sort_values('orders_per_100k', ascending=False)
    J['rows_out'] += put(st, 'gold', 'dm_state_market')
    J['rows_in'] = len(fs) + len(fx) + len(pop_uf)
st = save(st, 'lx_state_market')
mon = save(mon, 'lx_sales_monthly_usd')
COLS_ST = ['state', 'state_name', 'region_vn', 'population', 'orders', 'orders_per_100k', 'revenue_per_1000_brl', 'index_vs_brazil']
display(st[COLS_ST].round(1).head(8))
display(st[COLS_ST].round(1).tail(5))


def window(col, y):
    return v[v.year_month.between(f'{y}-01', f'{y}-08')][col].sum()


fxg = save(pd.DataFrame([dict(don_vi=u, t1_8_2017=window(c, 2017), t1_8_2018=window(c, 2018)) for u, c in [('BRL', 'price'), ('USD', 'price_usd')]])
           .assign(tang_truong=lambda x: x.t1_8_2018 / x.t1_8_2017 - 1), 'lx_growth_brl_usd')
rate = fx.assign(y=fx.date.dt.year, m=fx.date.dt.month)
rate = {y: rate[(rate.y == y) & (rate.m <= 8)].usd_brl.mean() for y in (2017, 2018)}
display(fxg.round(3))
print(f"Tỷ giá TB T1–8: 2017 = {rate[2017]:.3f}, 2018 = {rate[2018]:.3f} BRL/USD (BRL mất giá {rate[2018] / rate[2017] - 1:.0%}) "
      f"→ tăng trưởng đo bằng USD thấp hơn đo bằng BRL {fxg.tang_truong[0] - fxg.tang_truong[1]:.0%} điểm")

# %%
# Kiểm tra chất lượng tên thành phố bằng danh mục IBGE (khóa so khớp: tên không dấu + bang)
ibge_keys = set(pop_city.city_key)
ds_ = get('gold', 'dim_seller')
cities = pd.concat([dc[dc.customer_key > 0][['city', 'state']].assign(nguon='Khách hàng'),
                    ds_[ds_.seller_key > 0][['city', 'state']].assign(nguon='Người bán')])
cities['city_key'] = norm_text(cities.city) + '|' + cities.state
cities['khop_ibge'] = cities.city_key.isin(ibge_keys)
cq = save(cities.groupby('nguon').agg(so_dong=('city', 'size'), ten_khac_nhau=('city_key', 'nunique'), khop=('khop_ibge', 'sum'))
          .assign(ty_le_khop=lambda x: x.khop / x.so_dong).reset_index(), 'lx_city_quality')
display(cq)
miss = save(cities[~cities.khop_ibge].groupby(['nguon', 'city', 'state']).size().rename('so_dong').reset_index()
            .sort_values('so_dong', ascending=False), 'lx_city_unmatched')
print('Tên thành phố không có trong danh mục IBGE (nhiều nhất):')
display(miss.head(10))

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[1.7, 1]))
f2 = fx[fx.date.between('2017-01-01', '2018-08-31')]
ax[0].plot(f2.date, f2.usd_brl, color=BLUE, lw=1.6)
ax[0].set_title('Tỷ giá USD/BRL hằng ngày (API Ngân hàng Trung ương Brazil)', fontsize=11, fontweight='bold')
ax[0].set_ylabel('BRL cho 1 USD'); ax[0].grid(axis='y', color='#E5E5E5')
b = ax[1].bar(['Đo bằng BRL', 'Đo bằng USD'], fxg.tang_truong * 100, color=[BLUE, ORANGE], width=0.55)
ax[1].bar_label(b, labels=[f'+{x:.0f}%' for x in fxg.tang_truong * 100], fontsize=11, fontweight='bold')
ax[1].set_title('Tăng trưởng doanh thu T1–8/2018 so cùng kỳ', fontsize=11, fontweight='bold')
ax[1].set_ylim(0, fxg.tang_truong.max() * 118); ax[1].set_yticks([])
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(CHART / 'n6_fx_growth.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
s2 = st.sort_values('orders_per_100k')
REG_COLOR = {'Đông Nam': BLUE, 'Miền Nam': GREEN, 'Trung Tây': ORANGE, 'Đông Bắc': RED, 'Miền Bắc': GREY}
ax.barh(s2.state + ' – ' + s2.state_name, s2.orders_per_100k, color=s2.region_vn.map(REG_COLOR))
ax.axvline(nat, color='#333333', ls='--', lw=1)
ax.text(nat, 1, f' Cả nước: {nat:.0f} đơn / 100.000 dân', fontsize=9, va='center')
ax.legend(handles=[Patch(color=c, label=r_) for r_, c in REG_COLOR.items()], title='Vùng', frameon=False, loc='lower right')
ax.set_title('Số đơn trên 100.000 dân theo bang (01/2017–08/2018) – dân số lấy từ API IBGE', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(CHART / 'n6_state_market.png'); plt.close(fig)

# %% [markdown]
# ## 3. ĐƠN HÀNG DẠNG TÀI LIỆU JSON LỒNG NHAU (mô phỏng webhook của sàn)
# Thực tế sàn TMĐT gửi đơn hàng cho đối tác qua **API / webhook dạng JSON**: mỗi đơn là một tài liệu lồng nhau
# (khách hàng → địa chỉ, mảng sản phẩm, mảng thanh toán, mảng đánh giá). Bộ Olist công khai đã được "trải phẳng" thành 9 file CSV,
# nên job L17 **đóng gói lại 99.441 đơn thật thành tài liệu JSON** để mô phỏng nguồn này. **Giá trị là dữ liệu thật, chỉ hình dạng là mô phỏng.**
#
# Như ngoài đời, nguồn **đổi phiên bản API từ 01/2018**:
#
# | | API v1 (đơn đặt trước 2018) | API v2 (từ 01/2018) |
# |---|---|---|
# | Địa chỉ khách | `customer.zip`, `customer.city`, `customer.state` | **dời** vào `customer.address.{zip, city, state}` |
# | Mốc giao hàng | `approved_at`, `delivered_at`, `estimated_at` ở cấp ngoài | **gom** vào `delivery.{…}` và **thêm** `carrier_at` |
# | Sản phẩm | `items[].{item_no, product_id, seller_id, price, freight}` | **thêm** `items[].product.{category, weight_g}`; tiền gửi **dạng chữ kiểu Brazil** `"58,90"` |
#
# Thêm **3 dòng hỏng cố ý** (đánh dấu `TEST-`) vào file 03/2018 để kiểm thử cơ chế cách ly: JSON bị cắt cụt, máy chủ trả trang lỗi HTML, tài liệu thiếu `order_id`.
#
# ### L17 – Đóng gói 99.441 đơn thành tài liệu JSON (mỗi tháng một file JSON Lines)

# %%
so, sc, si, sp, sr, spd = (get('silver', n) for n in ['orders', 'customers', 'items', 'payments', 'reviews', 'products'])


def records(df, cols):
    """DataFrame -> list[dict] chuẩn JSON: ngày giờ thành chữ, rỗng thành null."""
    d = df[cols].copy()
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime('%Y-%m-%d %H:%M:%S')
    return d.astype(object).where(d.notna(), None).to_dict('records')


def group(df, key, cols):
    out = {}
    for k, r in zip(df[key], records(df, cols)):
        out.setdefault(k, []).append(r)
    return out


ITEMS = group(si.merge(spd[['product_id', 'category_pt', 'weight_g']], on='product_id', how='left').sort_values(['order_id', 'order_item_id']),
              'order_id', ['order_item_id', 'product_id', 'seller_id', 'price', 'freight_value', 'category_pt', 'weight_g'])
PAYS = group(sp.sort_values(['order_id', 'payment_sequential']), 'order_id', ['payment_sequential', 'payment_type', 'payment_installments', 'payment_value'])
REVS = group(sr.sort_values(['order_id', 'review_creation_date']), 'order_id', ['review_id', 'review_score', 'review_creation_date'])
CUST = dict(zip(sc.customer_id, records(sc, ['customer_unique_id', 'zip_prefix', 'city', 'state'])))
V2_FROM = '2018-01-01'
BRL_TXT = lambda x: None if x is None else f'{x:.2f}'.replace('.', ',')       # 58.9 -> "58,90"


def to_doc(o):
    v2, c, oid = o['order_purchase_timestamp'] >= V2_FROM, CUST[o['customer_id']], o['order_id']
    d = dict(api_version='v2' if v2 else 'v1', event='order.snapshot', order_id=oid, status=o['order_status'], purchased_at=o['order_purchase_timestamp'])
    if v2:
        d['customer'] = dict(id=o['customer_id'], unique_id=c['customer_unique_id'], address=dict(zip=c['zip_prefix'], city=c['city'], state=c['state']))
        d['delivery'] = dict(approved_at=o['order_approved_at'], carrier_at=o['order_delivered_carrier_date'],
                             delivered_at=o['order_delivered_customer_date'], estimated_at=o['order_estimated_delivery_date'])
        d['items'] = [dict(item_no=i['order_item_id'], product_id=i['product_id'], seller_id=i['seller_id'], price=BRL_TXT(i['price']),
                           freight=BRL_TXT(i['freight_value']), product=dict(category=i['category_pt'], weight_g=i['weight_g'])) for i in ITEMS.get(oid, [])]
    else:
        d['customer'] = dict(id=o['customer_id'], unique_id=c['customer_unique_id'], zip=c['zip_prefix'], city=c['city'], state=c['state'])
        d.update(approved_at=o['order_approved_at'], delivered_at=o['order_delivered_customer_date'], estimated_at=o['order_estimated_delivery_date'])
        d['items'] = [dict(item_no=i['order_item_id'], product_id=i['product_id'], seller_id=i['seller_id'], price=i['price'], freight=i['freight_value'])
                      for i in ITEMS.get(oid, [])]
    d['payments'] = [dict(seq=p['payment_sequential'], type=p['payment_type'], installments=p['payment_installments'], value=p['payment_value'])
                     for p in PAYS.get(oid, [])]
    d['reviews'] = [dict(review_id=r['review_id'], score=r['review_score'], created_at=r['review_creation_date']) for r in REVS.get(oid, [])]
    return d


JSON_DEFAULT = lambda o: o.item() if hasattr(o, 'item') else str(o)                 # số kiểu numpy -> số của Python
BAD_LINES = ['{"api_version": "v2", "event": "order.snapshot", "order_id": "TEST-BROKEN-1", "items": [{"item_no": 1, "price": "10,00"',
             '<html><body><h1>502 Bad Gateway</h1></body></html>',
             '{"api_version": "v2", "event": "order.snapshot", "status": "delivered", "items": [], "payments": [], "reviews": []}']
JDIR = LANDING / 'json' / 'orders'
with job('L17', 'Silver (9 bảng) → tài liệu JSON đơn hàng (landing)', 'Landing', 'silver/orders, customers, items, payments, reviews, products',
         'data/landing/json/orders/*.jsonl') as J:
    shutil.rmtree(JDIR, ignore_errors=True); JDIR.mkdir(parents=True)
    by_month = {}
    for o in records(so, ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                          'order_delivered_customer_date', 'order_estimated_delivery_date']):
        by_month.setdefault(o['order_purchase_timestamp'][:7], []).append(json.dumps(to_doc(o), ensure_ascii=False, default=JSON_DEFAULT))
    by_month['2018-03'] += BAD_LINES
    for m, lines in sorted(by_month.items()):
        (JDIR / f'orders_{m}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    J['rows_in'] = len(so); J['rows_out'] = sum(len(x) for x in by_month.values())
FILES_J = sorted(JDIR.glob('orders_*.jsonl'))
print(f'{len(FILES_J)} file JSON Lines, {sum(f.stat().st_size for f in FILES_J) / 1e6:.1f} MB')
for ver, m in [('v1', '2017-06'), ('v2', '2018-06')]:
    d = json.loads((JDIR / f'orders_{m}.jsonl').read_text(encoding='utf-8').splitlines()[0])
    print(f'\nVí dụ tài liệu API {ver}:\n' + json.dumps(d, ensure_ascii=False, indent=1)[:1100])

# %% [markdown]
# ### L18 – Bronze: mỗi file tháng là một lần nạp (một phiên bản Delta)
# Dòng JSON được lưu **nguyên văn** (kể cả dòng hỏng): Bronze là bản gốc để truy vết và nạp lại khi quy tắc xử lý thay đổi.

# %%
with job('L18', 'Landing JSONL → Bronze order_events_raw (nạp từng tháng)', 'Bronze', 'data/landing/json/orders/*.jsonl', 'bronze/order_events_raw') as J:
    for f in FILES_J:
        lines = f.read_text(encoding='utf-8').splitlines()
        df = pd.DataFrame(dict(file=f.name, line_no=np.arange(1, len(lines) + 1), raw_json=lines, _batch_id=BATCH, _load_ts=NOW()))
        J['rows_in'] += len(df); J['rows_out'] += put(df, 'bronze', 'order_events_raw', mode='append')
print('Số phiên bản của bronze/order_events_raw:', DeltaTable(str(LAKE / 'bronze' / 'order_events_raw')).version() + 1)

# %% [markdown]
# ### L19 – Silver: kiểm tra từng tài liệu, cách ly dòng lỗi, áp **hợp đồng dữ liệu** của từng phiên bản
# * **Kiểm tra**: đúng JSON, là đối tượng, có `order_id`, phiên bản API đã biết → không đạt thì ghi vào `order_events_quarantine` kèm lý do.
# * **Hợp đồng dữ liệu (data contract)**: mỗi phiên bản API có một lược đồ khai báo trước (kiểu của từng trường, kể cả trường lồng nhau).
# * Bảng `silver/order_events` giữ **nguyên cấu trúc lồng nhau** (cột struct / list) và được phép **mở rộng lược đồ** khi v2 xuất hiện.
# * Thử ghi lô v2 **khi chưa chuyển tiền dạng chữ `"58,90"` sang số**: Delta phải từ chối và **không ghi dở dang** (ACID).

# %%
TS = pa.timestamp('us')
MONEY = pa.float64()


def item_t(money):
    return pa.struct([('item_no', pa.int64()), ('product_id', pa.string()), ('seller_id', pa.string()), ('price', money), ('freight', money)])


PAY_T = pa.struct([('seq', pa.int64()), ('type', pa.string()), ('installments', pa.int64()), ('value', pa.float64())])
REV_T = pa.struct([('review_id', pa.string()), ('score', pa.int64()), ('created_at', TS)])
HEAD = [('api_version', pa.string()), ('event', pa.string()), ('order_id', pa.string()), ('status', pa.string()), ('purchased_at', TS)]
ADDR = [('zip', pa.string()), ('city', pa.string()), ('state', pa.string())]


def contract(ver, money=MONEY):
    if ver == 'v1':
        return pa.schema(HEAD + [('customer', pa.struct([('id', pa.string()), ('unique_id', pa.string())] + ADDR)),
                                 ('approved_at', TS), ('delivered_at', TS), ('estimated_at', TS),
                                 ('items', pa.list_(item_t(money))), ('payments', pa.list_(PAY_T)), ('reviews', pa.list_(REV_T))])
    item = pa.struct(list(item_t(money)) + [pa.field('product', pa.struct([('category', pa.string()), ('weight_g', pa.float64())]))])
    return pa.schema(HEAD + [('customer', pa.struct([('id', pa.string()), ('unique_id', pa.string()), ('address', pa.struct(ADDR))])),
                             ('delivery', pa.struct([('approved_at', TS), ('carrier_at', TS), ('delivered_at', TS), ('estimated_at', TS)])),
                             ('items', pa.list_(item)), ('payments', pa.list_(PAY_T)), ('reviews', pa.list_(REV_T))])


def check(line):
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        return None, f'JSON hỏng: {e.msg}'
    if not isinstance(d, dict):
        return None, 'Không phải đối tượng JSON'
    if not d.get('order_id'):
        return None, 'Thiếu trường bắt buộc order_id'
    if d.get('api_version') not in ('v1', 'v2'):
        return None, f"Phiên bản API chưa biết: {d.get('api_version')}"
    return d, None


tsv = lambda x: None if x is None else datetime.fromisoformat(x)
num = lambda x: None if x is None else float(x.replace(',', '.')) if isinstance(x, str) else float(x)


def normalize(d, fix_money=True):
    """Áp hợp đồng: chữ ngày giờ -> timestamp; (nếu fix_money) tiền kiểu Brazil '58,90' -> 58.9."""
    d = dict(d, purchased_at=tsv(d['purchased_at']))
    if d['api_version'] == 'v1':
        d.update({k: tsv(d.get(k)) for k in ('approved_at', 'delivered_at', 'estimated_at')})
    else:
        d['delivery'] = {k: tsv(x) for k, x in d['delivery'].items()}
    if fix_money:
        d['items'] = [dict(i, price=num(i['price']), freight=num(i['freight'])) for i in d['items']]
    d['reviews'] = [dict(r, created_at=tsv(r['created_at'])) for r in d['reviews']]
    return d


EV = str(LAKE / 'silver' / 'order_events')
bronze_ev = dset('bronze', 'order_events_raw')
QUAR, SV, ENFORCE = [], [], None
with job('L19', 'Bronze → Silver order_events (kiểm tra, hợp đồng dữ liệu, lược đồ mở rộng)', 'Silver', 'bronze/order_events_raw',
         'silver/order_events, order_events_quarantine') as J:
    for f in FILES_J:
        rows = bronze_ev.to_table(filter=pc.field('file') == f.name, columns=['line_no', 'raw_json']).to_pylist()
        good = {'v1': [], 'v2': []}
        for r in rows:
            d, err = check(r['raw_json'])
            if err:
                QUAR.append(dict(file=f.name, line_no=r['line_no'], reason=err, raw_head=r['raw_json'][:70]))
            else:
                good[d['api_version']].append(d)
        J['rows_in'] += len(rows)
        for ver, docs in good.items():
            if not docs:
                continue
            if ver == 'v2' and ENFORCE is None:                    # lần đầu gặp v2: thử ghi khi chưa xử lý tiền dạng chữ
                v_before = DeltaTable(EV).version()
                try:
                    write_deltalake(EV, pa.Table.from_pylist([normalize(x, fix_money=False) for x in docs], schema=contract('v2', pa.string())),
                                    mode='append', schema_mode='merge')
                    ENFORCE = dict(ket_qua='Delta chấp nhận (không mong đợi)')
                except Exception as e:
                    ENFORCE = dict(file=f.name, ket_qua='Delta từ chối cả lô', loi=str(e).split('\n')[-1][:160],
                                   phien_ban_truoc=v_before, phien_ban_sau=DeltaTable(EV).version())
            write_deltalake(EV, pa.Table.from_pylist([normalize(x) for x in docs], schema=contract(ver)), mode='append', schema_mode='merge')
            dt_ = DeltaTable(EV)
            SV.append(dict(version=dt_.version(), file=f.name, api_version=ver, rows=len(docs), leaf_fields=len(leaf_fields(dt_.to_pyarrow_dataset().schema))))
            J['rows_out'] += len(docs)
    quar = pd.DataFrame(QUAR)
    put(quar, 'silver', 'order_events_quarantine')
assert ENFORCE['ket_qua'] == 'Delta từ chối cả lô' and ENFORCE['phien_ban_truoc'] == ENFORCE['phien_ban_sau'], ENFORCE
print('Thử ghi lô v2 còn tiền dạng chữ "58,90":', ENFORCE)
display(save(quar, 'lx_quarantine'))
sv = save(pd.DataFrame(SV), 'lx_schema_versions')

# %%
# Lược đồ phiên bản 0 (chỉ có v1) và hiện tại: những trường nào được Delta tự thêm khi API v2 xuất hiện?
f0 = leaf_fields(DeltaTable(EV, version=0).to_pyarrow_dataset().schema)
f_now = leaf_fields(DeltaTable(EV).to_pyarrow_dataset().schema)
added = [x for x in f_now if x not in f0]
evo = save(pd.DataFrame(dict(truong_moi=added)), 'lx_schema_added')
print(f'Phiên bản 0: {len(f0)} trường lá · hiện tại: {len(f_now)} trường lá · thêm {len(added)} trường:', ', '.join(added))
print('Kiểu cột items:', DeltaTable(EV).to_pyarrow_dataset().schema.field('items').type)

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
ax.step(sv.version, sv.leaf_fields, where='post', color=BLUE, lw=2.2)
ax.scatter(sv.version, sv.leaf_fields, color=[ORANGE if x == 'v2' else BLUE for x in sv.api_version], zorder=3, s=22)
k = sv[sv.api_version == 'v2'].index[0]
ax.annotate(f'API v2 (01/2018): Delta tự thêm {len(added)} trường\n(customer.address, delivery, items[].product)', xy=(sv.version[k], sv.leaf_fields[k]),
            xytext=(sv.version[k] - 13, sv.leaf_fields[k] - 2.5), fontsize=10, arrowprops=dict(arrowstyle='->', color='#555555'))
ax.set_xticks(sv.version); ax.set_xticklabels(sv.file.str.slice(7, 14), rotation=60, fontsize=8)
ax.set_ylabel('Số trường lá trong lược đồ'); ax.set_ylim(min(sv.leaf_fields) - 4, max(sv.leaf_fields) + 3)
ax.set_title('Lược đồ bảng silver/order_events qua từng lần nạp (mỗi điểm = một phiên bản Delta)', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#EEEEEE')
fig.tight_layout(); fig.savefig(CHART / 'n6_schema_evolution.png'); plt.close(fig)

# %% [markdown]
# ### L20 – Làm phẳng bằng SQL (`UNNEST`) và đối soát với bảng Silver gốc
# Một câu SQL đọc được **cả hai phiên bản**: `COALESCE(customer.address.state, customer.state)` lấy địa chỉ ở chỗ mới, nếu rỗng thì lấy chỗ cũ.
# Kết quả làm phẳng phải **khớp tuyệt đối** với `silver/items`, `silver/payments` (cùng số dòng, cùng tổng tiền) – chứng minh chuỗi
# đóng gói JSON → nạp → kiểm tra → làm phẳng không mất hay sai một bản ghi nào (ngoài 3 dòng hỏng cố ý).

# %%
ev = dset('silver', 'order_events')
SQL_ITEMS = """
WITH x AS (SELECT order_id, api_version, purchased_at, COALESCE(customer.address.state, customer.state) AS customer_state,
                  COALESCE(customer.address.city, customer.city) AS customer_city, COALESCE(delivery.delivered_at, delivered_at) AS delivered_at,
                  UNNEST(items) AS it FROM ev)
SELECT order_id, api_version, purchased_at, customer_state, customer_city, delivered_at, it.item_no, it.product_id, it.seller_id,
       it.price, it.freight, it.product.category AS category_pt, it.product.weight_g AS weight_g FROM x"""
SQL_PAYS = """WITH x AS (SELECT order_id, api_version, UNNEST(payments) AS p FROM ev)
              SELECT order_id, api_version, p.seq, p.type, p.installments, p.value FROM x"""
with job('L20', 'Silver lồng nhau → Silver phẳng (DuckDB UNNEST) + đối soát', 'Silver', 'silver/order_events', 'silver/order_events_items, order_events_payments') as J:
    flat_i, flat_p = duckdb.sql(SQL_ITEMS).df(), duckdb.sql(SQL_PAYS).df()
    J['rows_in'] = ev.count_rows()
    J['rows_out'] = put(flat_i, 'silver', 'order_events_items') + put(flat_p, 'silver', 'order_events_payments')

st_map = so[['order_id', 'customer_id']].merge(sc[['customer_id', 'state']], on='customer_id').set_index('order_id').state
st_json = flat_i.drop_duplicates('order_id').set_index('order_id').customer_state
REC = [('Số đơn (tài liệu hợp lệ)', len(so), ev.count_rows()),
       ('Số dòng sản phẩm', len(si), len(flat_i)),
       ('Tổng tiền hàng (BRL)', round(si.price.sum(), 2), round(flat_i.price.sum(), 2)),
       ('Tổng phí vận chuyển (BRL)', round(si.freight_value.sum(), 2), round(flat_i.freight.sum(), 2)),
       ('Số dòng thanh toán', len(sp), len(flat_p)),
       ('Tổng tiền thanh toán (BRL)', round(sp.payment_value.sum(), 2), round(flat_p.value.sum(), 2)),
       ('Đơn có bang khách khác với silver/customers', 0, int((st_json != st_map.reindex(st_json.index)).sum())),
       ('Dòng bị cách ly (3 dòng hỏng cố ý)', len(BAD_LINES), len(quar))]
rec = save(pd.DataFrame(REC, columns=['chi_so', 'silver_goc', 'tu_json']).assign(khop=lambda x: np.isclose(x.silver_goc.astype(float), x.tu_json.astype(float))),
           'lx_json_reconcile')
display(rec)
assert rec.khop.all(), 'đối soát JSON không khớp'
display(flat_i.groupby('api_version').agg(dong=('order_id', 'size'), co_nganh_hang=('category_pt', 'count'), co_can_nang=('weight_g', 'count')))

# %% [markdown]
# ### So sánh ba cách trả lời cùng một câu hỏi: *doanh thu (tiền hàng) theo bang của khách, đơn đặt năm 2018*
# 1. **Bronze – JSON thô**: phân tích chuỗi JSON mỗi lần hỏi (schema-on-read): linh hoạt nhất, chậm nhất.
# 2. **Silver – cột lồng nhau** (Parquet): đã có kiểu, đọc theo cột, chỉ cần `UNNEST`.
# 3. **Gold – mô hình sao** (`fact_sales` ⋈ `dim_customer`): đã làm phẳng, nhanh nhất cho BI.

# %%
braw = dset('bronze', 'order_events_raw')
fsales, dgeo = dset('gold', 'fact_sales'), dset('gold', 'dim_geography')
Q = {'Bronze – chuỗi JSON thô': """
        SELECT COALESCE(json_extract_string(raw_json, '$.customer.address.state'), json_extract_string(raw_json, '$.customer.state')) AS state,
               SUM(list_sum(list_transform(json_extract_string(raw_json, '$.items[*].price'), lambda x: CAST(replace(x, ',', '.') AS DOUBLE)))) AS revenue
        FROM braw WHERE json_valid(raw_json) AND json_extract_string(raw_json, '$.order_id') IS NOT NULL
              AND json_extract_string(raw_json, '$.purchased_at') >= '2018-01-01' GROUP BY 1""",
     'Silver – cột lồng nhau (struct / list)': """
        WITH x AS (SELECT COALESCE(customer.address.state, customer.state) AS state, UNNEST(items) AS it FROM ev WHERE purchased_at >= TIMESTAMP '2018-01-01')
        SELECT state, SUM(it.price) AS revenue FROM x GROUP BY 1""",
     'Gold – mô hình sao': """
        SELECT g.state, SUM(f.price) AS revenue FROM fsales f JOIN dgeo g ON f.customer_geo_key = g.geo_key WHERE f.date_key >= 20180101 GROUP BY 1"""}
BM, ref = [], None
for name, sql in Q.items():
    out, ms = bench(lambda: duckdb.sql(sql).df())
    out = out.set_index('state').revenue.sort_index()
    ref = out if ref is None else ref
    BM.append(dict(cach=name, ms=ms, so_bang=len(out), tong_doanh_thu=round(out.sum(), 2), khop_ket_qua=bool(np.allclose(out.values, ref.reindex(out.index).values))))
bm = save(pd.DataFrame(BM), 'lx_query_bench')
display(bm)
assert bm.khop_ket_qua.all()
size = save(pd.DataFrame([dict(vung='Landing – file JSON Lines', mb=round(sum(f.stat().st_size for f in FILES_J) / 1e6, 1)),
                          dict(vung='Bronze – chuỗi JSON (Delta, nén)', mb=round(folder_mb(LAKE / 'bronze' / 'order_events_raw'), 1)),
                          dict(vung='Silver – cột lồng nhau (Delta)', mb=round(folder_mb(LAKE / 'silver' / 'order_events'), 1)),
                          dict(vung='Silver – 2 bảng phẳng (Delta)', mb=round(folder_mb(LAKE / 'silver' / 'order_events_items') +
                                                                             folder_mb(LAKE / 'silver' / 'order_events_payments'), 1))]), 'lx_json_storage')
display(size)

# %% [markdown]
# ## 4. VĂN BẢN ĐÁNH GIÁ – DỮ LIỆU KHÔNG CẤU TRÚC
# ### L21 – Silver: làm sạch văn bản và che dữ liệu cá nhân
# | Bước | Lý do |
# |---|---|
# | Ghép tiêu đề + nội dung, chuẩn hóa Unicode (NFC), bỏ xuống dòng / tab, gộp khoảng trắng | Bình luận gốc có ký tự xuống dòng làm vỡ file CSV, cùng chữ nhưng mã Unicode khác |
# | **Che dữ liệu cá nhân**: email, đường link, số CPF, số điện thoại, dãy số ≥ 6 chữ số (số hóa đơn) | Khách tự gõ số điện thoại vào bình luận; Silver dùng cho phân tích nên **không được chứa** dữ liệu cá nhân (bản gốc chỉ còn ở Bronze, quyền truy cập hạn chế) |
# | Đặc trưng: số chữ, số từ, emoji, dấu "!", tỷ lệ chữ in hoa, câu ngắn chung chung, mã băm để phát hiện câu trùng | Dùng cho phân tích và mô hình AI ở N7 |

# %%
PII = [('[EMAIL]', r'[\w.+-]+@[\w-]+\.[\w.]+'), ('[URL]', r'(?:https?://|www\.)\S+'), ('[CPF]', r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'),
       ('[PHONE]', r'\(?\b\d{2}\)?[\s.-]?9?\d{4}[\s.-]?\d{4}\b'), ('[DOC_NO]', r'\b\d{6,}\b')]
EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿]')
with job('L21', 'Bronze order_reviews → Silver review_text (làm sạch, che dữ liệu cá nhân)', 'Silver', 'bronze/order_reviews', 'silver/review_text') as J:
    br = get('bronze', 'order_reviews')
    J['rows_in'] = len(br)
    t = br[['review_id', 'order_id', 'review_score', 'review_creation_date']].copy()
    title, msg = br.review_comment_title.fillna('').str.strip(), br.review_comment_message.fillna('').str.strip()
    t['text_raw'] = (title + '. ' + msg).str.strip('. ').where(title != '', msg)
    t = t[t.text_raw != ''].copy()
    t['line_breaks'] = t.text_raw.str.count(r'[\r\n]+')
    t['n_emoji'] = t.text_raw.map(lambda x: len(EMOJI.findall(x)))
    clean = t.text_raw.map(lambda x: unicodedata.normalize('NFC', x)).str.replace(r'[\r\n\t]+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    masked, PII_N = clean, {}
    for tag, p in PII:
        PII_N[tag] = int(masked.str.count(p).sum())
        masked = masked.str.replace(p, tag, regex=True)
    t['text_masked'] = masked
    t['has_pii'] = (masked != clean).astype(int)
    t['text_norm'] = norm_text(masked)
    t['n_chars'] = masked.str.len()
    t['n_words'] = t.text_norm.str.split().str.len()
    t['n_exclaim'] = masked.str.count('!')
    letters = clean.str.count(r'[A-Za-zÀ-ÿ]')
    t['caps_ratio'] = (clean.str.count(r'[A-ZÀ-Þ]') / letters.where(letters > 0)).fillna(0).round(3)
    t['is_short'] = (t.n_words <= 2).astype(int)
    t['text_hash'] = t.text_norm.map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:16])
    t['dup_count'] = t.groupby('text_hash').text_hash.transform('size')
    t['review_score'] = t.review_score.astype(int)
    t['review_date'] = pd.to_datetime(t.review_creation_date)
    t = t.drop(columns=['text_raw', 'review_creation_date'])        # bản gốc (còn dữ liệu cá nhân) chỉ nằm ở Bronze
    J['rows_out'] = put(t, 'silver', 'review_text')
assert not t.text_masked.str.contains(PII[3][1], regex=True).any(), 'Silver còn số điện thoại'
txt = save(pd.DataFrame([
    dict(chi_so='Đánh giá có chữ (tiêu đề hoặc nội dung)', gia_tri=len(t)),
    dict(chi_so='Có ký tự xuống dòng (đã gộp)', gia_tri=int((t.line_breaks > 0).sum())),
    dict(chi_so='Có emoji', gia_tri=int((t.n_emoji > 0).sum())),
    dict(chi_so='Câu ngắn ≤ 2 từ ("bom", "ótimo"...)', gia_tri=int(t.is_short.sum())),
    dict(chi_so='Câu trùng nguyên văn với đánh giá khác', gia_tri=int((t.dup_count > 1).sum()))] +
    [dict(chi_so=f'Đã che {tag}', gia_tri=n) for tag, n in PII_N.items()]), 'lx_text_profile')
display(txt)
print('Ví dụ sau khi che (Silver):')
display(t[t.has_pii == 1][['review_score', 'text_masked']].head(6))

# %% [markdown]
# ### L22 – Gold: nối văn bản với bảng sự kiện và tạo chỉ mục từ
# * `fact_review_text`: mỗi bình luận gắn khóa của `fact_review` / `fact_order` → hỏi được "khách nói gì ở những đơn giao trễ?".
# * `review_token_index`: **chỉ mục ngược** (từ → bình luận) – cách các công cụ tìm kiếm biến văn bản tự do thành dữ liệu tra cứu được bằng SQL.

# %%
with job('L22', 'Silver review_text → Gold fact_review_text + review_token_index', 'Gold', 'silver/review_text; gold/fact_review, fact_order',
         'gold/fact_review_text, review_token_index') as J:
    fr = get('gold', 'fact_review')[['review_key', 'review_id', 'order_id', 'date_key', 'customer_key', 'delivery_status', 'is_late']]
    fo = get('gold', 'fact_order')[['order_key', 'order_id', 'delay_days', 'is_delivered', 'is_canceled']]
    frt = t.merge(fr, on=['review_id', 'order_id'], how='inner').merge(fo, on='order_id', how='left')
    frt = frt[['review_key', 'order_key', 'date_key', 'customer_key', 'review_score', 'delivery_status', 'is_late', 'delay_days', 'is_delivered',
               'is_canceled', 'n_chars', 'n_words', 'n_emoji', 'n_exclaim', 'caps_ratio', 'has_pii', 'is_short', 'dup_count', 'text_masked', 'text_norm']]
    tok = frt[['review_key', 'text_norm']].assign(token=lambda x: x.text_norm.str.split()).explode('token')
    tok = tok[tok.token.notna() & ~tok.token.isin(STOP_PT) & (tok.token.str.len() >= 2)]
    idx = tok.groupby(['token', 'review_key']).size().rename('tf').reset_index()
    idx['df'] = idx.groupby('token').review_key.transform('size')
    J['rows_in'] = len(t)
    J['rows_out'] = put(frt, 'gold', 'fact_review_text') + put(idx, 'gold', 'review_token_index')
assert len(frt) == len(t), 'có bình luận không nối được với fact_review'
print(f'{idx.token.nunique():,} từ khác nhau · {len(idx):,} cặp (từ, bình luận)')

# %%
# Tìm kiếm bằng chỉ mục: bình luận có nhắc tới giao hàng ở đơn đúng hạn và trễ hạn
frt_ = dset('gold', 'fact_review_text'); idx_ = dset('gold', 'review_token_index')
DELIVERY_WORDS = ['entrega', 'entregue', 'entregar', 'entregou', 'prazo', 'atraso', 'atrasado', 'atrasou', 'demora', 'demorou', 'chegou', 'chegar',
                  'recebi', 'receber', 'recebido', 'correio', 'correios', 'transportadora']
words_sql = ', '.join(f"'{w}'" for w in DELIVERY_WORDS)
mention = save(duckdb.sql(f"""
    WITH hit AS (SELECT DISTINCT review_key FROM idx_ WHERE token IN ({words_sql}))
    SELECT f.delivery_status AS trang_thai_giao, COUNT(*) AS so_binh_luan,
           ROUND(100.0 * AVG(CASE WHEN h.review_key IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_nhac_giao_hang,
           ROUND(AVG(f.review_score), 2) AS diem_tb
    FROM frt_ f LEFT JOIN hit h USING (review_key) GROUP BY 1 ORDER BY 2 DESC""").df(), 'lx_text_delivery_mention')
display(mention)

# Đơn hệ thống ghi "đã giao" nhưng khách viết "chưa nhận được" → tranh chấp giao hàng cần kiểm tra
dispute = save(duckdb.sql(r"""
    SELECT delivery_status AS trang_thai_giao, COUNT(*) AS so_don, ROUND(AVG(review_score), 2) AS diem_tb
    FROM frt_ WHERE is_delivered = 1 AND regexp_matches(text_norm, '\b(nao|ainda nao) (recebi|chegou|foi entregue|recebemos)\b')
    GROUP BY 1 ORDER BY 2 DESC""").df(), 'lx_text_dispute')
display(dispute)

fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
m2 = mention[mention.trang_thai_giao.isin(['Đúng hạn', 'Trễ hạn'])].set_index('trang_thai_giao').loc[['Đúng hạn', 'Trễ hạn']]
bars = ax.bar(m2.index, m2.pct_nhac_giao_hang, color=[GREEN, RED], width=0.5)
ax.bar_label(bars, labels=[f'{p:.0f}% bình luận\nđiểm TB {s:.1f}'.replace('.', ',') for p, s in zip(m2.pct_nhac_giao_hang, m2.diem_tb)], fontsize=10)
ax.set_ylim(0, m2.pct_nhac_giao_hang.max() * 1.35); ax.set_yticks([])
ax.set_title('Tỷ lệ bình luận nhắc tới giao hàng (tìm qua chỉ mục từ)', fontsize=12, fontweight='bold')
ax.spines[['top', 'right', 'left']].set_visible(False)
fig.tight_layout(); fig.savefig(CHART / 'n6_text_delivery.png'); plt.close(fig)

# %% [markdown]
# ### Publish: ảnh chụp Parquet của các bảng Gold mới cho Power BI (cùng cách job L12 của N5)

# %%
for n in ['dm_sales_monthly_usd', 'dm_state_market', 'fact_review_text']:
    pq.write_table(DeltaTable(str(LAKE / 'gold' / n)).to_pyarrow_table(), LAKE / 'publish' / f'{n}.parquet')
    print(f'publish/{n}.parquet: {pq.ParquetFile(LAKE / "publish" / f"{n}.parquet").metadata.num_rows:,} dòng')

# %% [markdown]
# ## 5. TỔNG KẾT: VẤN ĐỀ → GIẢI PHÁP TRONG LAKEHOUSE → BẰNG CHỨNG

# %%
lay = []
for t_ in N6_TABLES:
    dt_ = DeltaTable(str(LAKE / t_))
    lay.append(dict(table=t_, rows=dt_.to_pyarrow_dataset().count_rows(), cols=len(dt_.schema().fields), versions=dt_.version() + 1,
                    mb=round(folder_mb(LAKE / t_), 2)))
lay = save(pd.DataFrame(lay), 'lx_layers')
display(lay)
b_ms = bm.set_index('cach').ms
CAP = [
    ('Bán cấu trúc', 'Mỗi API một dạng JSON, số / ngày dạng chữ', 'Landing giữ file gốc; Bronze lưu nguyên văn; Silver ép kiểu theo từng nguồn',
     f"{len(MANI)} phản hồi, {int(mani.records.sum()):,} bản ghi từ 3 API thật"),
    ('Bán cấu trúc', 'Nạp lại cùng file gây trùng dữ liệu', 'MERGE theo mã băm file + số thứ tự bản ghi (lũy đẳng)',
     f'Nạp lần 2: thêm {sum(again)} dòng'),
    ('Bán cấu trúc', 'Dòng tiêu đề lẫn trong dữ liệu, giá trị thiếu "..."', 'Nhận diện theo nội dung, chuyển thành rỗng',
     f'{head_uf + head_city} dòng tiêu đề bỏ, {miss_uf + miss_city} giá trị thiếu'),
    ('Bán cấu trúc', 'JSON lồng nhau (đối tượng trong đối tượng, mảng)', 'Delta lưu cột struct / list; DuckDB UNNEST làm phẳng',
     f'{len(flat_i):,} dòng sản phẩm, {len(flat_p):,} dòng thanh toán khớp 100% bảng gốc'),
    ('Bán cấu trúc', 'API đổi phiên bản: thêm / dời trường', 'Schema evolution (mergeSchema) + COALESCE hai vị trí',
     f'Lược đồ tự thêm {len(added)} trường, không dựng lại bảng'),
    ('Bán cấu trúc', 'Kiểu dữ liệu đổi ("58,90")', 'Schema enforcement + ACID: Delta từ chối cả lô, không ghi dở',
     f"Phiên bản trước / sau lần ghi lỗi: {ENFORCE['phien_ban_truoc']} / {ENFORCE['phien_ban_sau']}"),
    ('Bán cấu trúc', 'Dòng JSON hỏng làm dừng pipeline', 'Kiểm tra từng dòng, cách ly vào bảng quarantine kèm lý do', f'{len(quar)} dòng cách ly, pipeline chạy tiếp'),
    ('Bán cấu trúc', 'Muốn hỏi ngay khi chưa mô hình hóa', 'Schema-on-read: SQL trên chuỗi JSON thô',
     f"Cùng kết quả; Bronze {b_ms.iloc[0]:.0f} ms, Silver {b_ms.iloc[1]:.0f} ms, Gold {b_ms.iloc[2]:.0f} ms"),
    ('Không cấu trúc', 'Văn bản tự do, xuống dòng, emoji', 'Chuẩn hóa Unicode, gộp khoảng trắng, đặc trưng văn bản',
     f"{len(t):,} bình luận; {int((t.line_breaks > 0).sum()):,} có xuống dòng"),
    ('Không cấu trúc', 'Dữ liệu cá nhân trong bình luận', 'Che ở Silver; bản gốc chỉ ở Bronze', f"Che {sum(PII_N.values())} chuỗi (số điện thoại, link, số hóa đơn)"),
    ('Không cấu trúc', 'Không tìm / nối được với bảng sự kiện', 'Chỉ mục từ + khóa của fact_review / fact_order',
     f"{idx.token.nunique():,} từ; đơn trễ nhắc giao hàng {m2.pct_nhac_giao_hang['Trễ hạn']:.0f}% so với {m2.pct_nhac_giao_hang['Đúng hạn']:.0f}% đơn đúng hạn"),
    ('Tích hợp', 'Thiếu ngữ cảnh bên ngoài (tỷ giá, dân số)', 'Nối dữ liệu API vào Gold', f"Tăng trưởng T1–8/2018: BRL +{fxg.tang_truong[0]:.0%}, USD +{fxg.tang_truong[1]:.0%}")]
cap = save(pd.DataFrame(CAP, columns=['loai_du_lieu', 'van_de', 'giai_phap_lakehouse', 'bang_chung']), 'lx_capabilities')
display(cap)
save(pd.DataFrame(LOG), 'lx_log')
display(pd.DataFrame(LOG))
