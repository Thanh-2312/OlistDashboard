"""Sinh dữ liệu cho trang web phần mở rộng (05_Web_Dashboard/mo_rong.html) và chép mã nguồn / notebook / biểu đồ /
mẫu dữ liệu vào 05_Web_Dashboard/mo_rong/ để đưa lên GitHub.

Chạy sau N6, N7:  python _build/web_mo_rong.py   ->  05_Web_Dashboard/mo_rong_data.js + 05_Web_Dashboard/mo_rong/
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'notebooks'))
from olist_utils import STOP_PT, norm_text  # noqa: E402

WEB, RES, LAKE = ROOT / '05_Web_Dashboard', ROOT / 'data' / 'results', ROOT / 'data' / 'lakehouse'
OUT = WEB / 'mo_rong'
R = lambda n: pd.read_csv(RES / f'{n}.csv')
G = lambda layer, n: DeltaTable(str(LAKE / layer / n)).to_pandas()
rows = lambda df: json.loads(df.to_json(orient='values', force_ascii=False))


# ------------------------------------------------------------------ 1. mô hình cảm xúc chạy được trong trình duyệt
rt = G('silver', 'review_text')
lab = rt[rt.review_score != 3]
STOP = sorted(STOP_PT)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words=STOP)
X = vec.fit_transform(lab.text_norm)
lr = LogisticRegression(C=4, max_iter=3000, class_weight='balanced').fit(X, (lab.review_score <= 2).astype(int))
VOC = {t: [round(float(vec.idf_[i]), 5), round(float(lr.coef_[0][i]), 5)] for t, i in vec.vocabulary_.items()}
B0 = round(float(lr.intercept_[0]), 5)


def p_neg(text):
    """Cài lại TF-IDF + hồi quy logistic bằng Python thuần – đúng các bước mà mo_rong.html làm bằng JavaScript."""
    toks = [w for w in norm_text(pd.Series([text]))[0].split() if len(w) >= 2 and w not in STOP_PT]
    grams = toks + [f'{a} {b}' for a, b in zip(toks, toks[1:])]
    cnt = {}
    for g in grams:
        if g in VOC:
            cnt[g] = cnt.get(g, 0) + 1
    w = {g: (1 + math.log(c)) * VOC[g][0] for g, c in cnt.items()}
    nrm = math.sqrt(sum(x * x for x in w.values())) or 1
    z = B0 + sum(VOC[g][1] * x / nrm for g, x in w.items())
    return 1 / (1 + math.exp(-z))


samp = lab.sample(2000, random_state=1)
diff = np.abs(np.array([p_neg(t) for t in samp.text_masked]) - lr.predict_proba(vec.transform(samp.text_norm))[:, 1]).max()
assert diff < 1e-3, f'bản cài lại lệch {diff}'
EXAMPLES = [('Chegou rápido, produto excelente, recomendo!', 'Hàng đến nhanh, sản phẩm tuyệt vời, khuyên mua!'),
            ('Não recebi o produto até agora, péssimo.', 'Đến giờ vẫn chưa nhận được hàng, tệ hại.'),
            ('Produto bom, mas a entrega demorou muito.', 'Sản phẩm tốt nhưng giao hàng quá lâu.'),
            ('Veio quebrado e ninguém responde meu contato.', 'Hàng bị vỡ và không ai trả lời liên hệ của tôi.')]
print(f'Mô hình web: {len(VOC):,} từ / cụm từ, lệch tối đa so với scikit-learn {diff:.1e}')
for t, _ in EXAMPLES:
    print(f'  {p_neg(t):.4f}  {t}')

# ------------------------------------------------------------------ 2. dữ liệu cho các biểu đồ
fx = G('silver', 'fx_usd_brl').sort_values('date')
rate = {y: round(float(fx[(fx.date.dt.year == y) & (fx.date.dt.month <= 8)].usd_brl.mean()), 3) for y in (2017, 2018)}
mon = G('gold', 'dm_sales_monthly_usd').sort_values('year_month')
mon = mon[mon.year_month.between('2017-01', '2018-08')]
st = R('lx_state_market')
trend = pd.read_csv(RES / 'ai_theme_trend.csv', index_col=0)
JDIR = ROOT / 'data' / 'landing' / 'json' / 'orders'
first = lambda m: json.loads((JDIR / f'orders_{m}.jsonl').read_text(encoding='utf-8').splitlines()[0])
MR = dict(
    caps=rows(R('lx_capabilities')),
    api=rows(R('lx_api_manifest')[['source', 'dataset', 'description', 'records', 'kb', 'url']]),
    quirks=rows(R('lx_api_quirks')),
    idem=rows(R('lx_idempotent')),
    fx=dict(d=fx.date.dt.strftime('%Y-%m-%d').tolist(), r=fx.usd_brl.round(4).tolist(), filled=fx.is_filled.tolist()),
    rate=rate,
    growth=rows(R('lx_growth_brl_usd')),
    monthly=rows(mon[['year_month', 'revenue_brl', 'revenue_usd']].round(0)),
    states=rows(st[['state', 'state_name', 'region_vn', 'population', 'customers', 'orders', 'orders_per_100k', 'revenue_per_1000_brl', 'index_vs_brazil']].round(2)),
    nat=round(float(st.orders.sum() / st.population.sum() * 1e5), 1),
    city=rows(R('lx_city_quality')), unmatched=rows(R('lx_city_unmatched').head(12)),
    holiday=R('lx_holiday_check').ket_qua.value_counts().to_dict(),
    sv=rows(R('lx_schema_versions')), added=R('lx_schema_added').truong_moi.tolist(),
    v1=json.dumps(first('2017-06'), ensure_ascii=False, indent=2), v2=json.dumps(first('2018-06'), ensure_ascii=False, indent=2),
    quar=rows(R('lx_quarantine')), rec=rows(R('lx_json_reconcile')), bench=rows(R('lx_query_bench')), storage=rows(R('lx_json_storage')),
    tprofile=rows(R('lx_text_profile')), mention=rows(R('lx_text_delivery_mention')), dispute=rows(R('lx_text_dispute')),
    ai=dict(metrics=rows(R('ai_sentiment_metrics')), byscore=rows(R('ai_sentiment_by_score')), mismatch=rows(R('ai_sentiment_mismatch_examples')),
            words=rows(R('ai_sentiment_words')), topics=rows(R('ai_topics')), themes=rows(R('ai_themes')),
            trend=dict(months=trend.index.tolist(), series={c: trend[c].tolist() for c in trend.columns}),
            late=rows(R('ai_late_metrics')), lift=rows(R('ai_late_lift')), imp=rows(R('ai_late_importance')),
            watch=rows(R('ai_transit_watchlist')[['order_id', 'risk_score', 'customer_state', 'seller_state', 'days_left']]),
            summary=rows(R('ai_summary'))),
    model=dict(vocab=VOC, b=B0, stop=STOP, n=len(lab), examples=[[t, vn, round(p_neg(t), 4)] for t, vn in EXAMPLES]))
js = 'window.MR = ' + json.dumps(MR, ensure_ascii=False, separators=(',', ':')) + ';\n'
(WEB / 'mo_rong_data.js').write_text(js, encoding='utf-8')
print(f'mo_rong_data.js: {len(js.encode()) / 1e6:.2f} MB')

# ------------------------------------------------------------------ 3. chép mã nguồn, notebook, biểu đồ, mẫu dữ liệu vào repo
for sub in ['src', 'notebooks', 'charts', 'du_lieu_mau']:
    shutil.rmtree(OUT / sub, ignore_errors=True)
    (OUT / sub).mkdir(parents=True)
for f in ['_build/nb/N6.py', '_build/nb/N7.py', 'notebooks/olist_utils.py', '_build/make_notebooks.py', '_build/web_mo_rong.py']:
    shutil.copy(ROOT / f, OUT / 'src')
for f in ['N6_Lakehouse_ban_cau_truc_khong_cau_truc.ipynb', 'N7_AI_khai_thac_du_lieu.ipynb']:
    shutil.copy(ROOT / 'notebooks' / f, OUT / 'notebooks')
for f in sorted((ROOT / 'assets' / 'charts').glob('n[67]_*.png')):
    shutil.copy(f, OUT / 'charts')
for f in sorted((ROOT / 'data' / 'landing' / 'api').rglob('*.json')):
    shutil.copy(f, OUT / 'du_lieu_mau' / f'{f.parent.name}__{f.name}')
for m, n in [('2017-06', 40), ('2018-03', None)]:          # 40 dòng v1; file 03/2018 lấy 40 dòng đầu + 3 dòng hỏng cố ý ở cuối
    ls = (JDIR / f'orders_{m}.jsonl').read_text(encoding='utf-8').splitlines()
    ls = ls[:40] + (ls[-3:] if n is None else [])
    (OUT / 'du_lieu_mau' / f'orders_{m}_mau.jsonl').write_text('\n'.join(ls) + '\n', encoding='utf-8')
print('Đã chép vào', OUT, ':', sum(1 for _ in OUT.rglob('*') if _.is_file()), 'file')
