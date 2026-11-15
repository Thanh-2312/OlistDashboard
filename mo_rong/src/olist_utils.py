"""Hằng số và hàm dùng chung cho 4 notebook N1–N4 (bộ dữ liệu Olist).

Đặt ở một chỗ để mọi notebook dùng cùng một quy ước: đường dẫn, tên bảng, bảng vùng miền,
nhóm ngành hàng, tên tiếng Việt, và vài hàm tiện ích (bỏ dấu, khoảng cách Haversine...).
"""
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- đường dẫn
HERE = Path(__file__).resolve().parent            # .../BTL_Cuoi_Ki/notebooks
PROJECT = HERE.parent                             # .../BTL_Cuoi_Ki
RAW_DIR = PROJECT.parent                          # .../archive  (9 file CSV gốc của Olist)
DATA = PROJECT / 'data'
RES = DATA / 'results'                            # kết quả phân tích -> dùng để dựng file Excel
DW_CSV = DATA / 'dw'                              # bảng kho dữ liệu xuất CSV -> Power Query nạp vào Excel
STG_DB_PATH = DATA / 'olist_staging.db'           # CSDL SQLite lớp Staging (stg_* thô, cln_* đã làm sạch)
DB_PATH = DATA / 'olist_dw.db'                    # CSDL SQLite lớp Data Warehouse + Data Mart + Metadata
for _p in (RES, DW_CSV):
    _p.mkdir(parents=True, exist_ok=True)

FILES = dict(customers='olist_customers_dataset.csv', geolocation='olist_geolocation_dataset.csv',
             orders='olist_orders_dataset.csv', items='olist_order_items_dataset.csv',
             payments='olist_order_payments_dataset.csv', reviews='olist_order_reviews_dataset.csv',
             products='olist_products_dataset.csv', sellers='olist_sellers_dataset.csv',
             translation='product_category_name_translation.csv')
TABLE_VN = dict(customers='Khách hàng', geolocation='Tọa độ mã bưu chính', orders='Đơn hàng', items='Chi tiết đơn hàng',
                payments='Thanh toán', reviews='Đánh giá', products='Sản phẩm', sellers='Người bán', translation='Dịch tên ngành hàng')

# Cột số / ngày giờ của từng bảng (dữ liệu gốc đều là text trong CSV)
NUM = dict(geolocation=['geolocation_lat', 'geolocation_lng'],
           items=['order_item_id', 'price', 'freight_value'],
           payments=['payment_sequential', 'payment_installments', 'payment_value'],
           reviews=['review_score'],
           products=['product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                     'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'])
DT = dict(orders=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                  'order_delivered_customer_date', 'order_estimated_delivery_date'],
          items=['shipping_limit_date'], reviews=['review_creation_date', 'review_answer_timestamp'])

# Giai đoạn đủ dữ liệu dùng cho biểu đồ xu hướng (2016 và 09–10/2018 quá thưa)
TREND_START, TREND_END = '2017-01-01', '2018-08-31'


def load_raw():
    """Đọc 9 file CSV dạng text (dtype=str) để không mất số 0 đầu của mã bưu chính."""
    return {k: pd.read_csv(RAW_DIR / f, dtype=str) for k, f in FILES.items()}


def to_typed(raw):
    """Bản sao đã ép kiểu số / ngày giờ (chưa làm sạch) – dùng để kiểm tra lỗi & thống kê."""
    T = {k: v.copy() for k, v in raw.items()}
    for t, cols in NUM.items():
        for c in cols:
            T[t][c] = pd.to_numeric(T[t][c])
    for t, cols in DT.items():
        for c in cols:
            T[t][c] = pd.to_datetime(T[t][c])
    return T


def save(df, name, index=False):
    """Lưu kết quả vào data/results/<name>.csv (UTF-8)."""
    df.to_csv(RES / f'{name}.csv', index=index, encoding='utf-8')
    return df


# ---------------------------------------------------------------- địa lý Brazil
STATE_NAME = dict(AC='Acre', AL='Alagoas', AP='Amapá', AM='Amazonas', BA='Bahia', CE='Ceará', DF='Distrito Federal',
                  ES='Espírito Santo', GO='Goiás', MA='Maranhão', MT='Mato Grosso', MS='Mato Grosso do Sul',
                  MG='Minas Gerais', PA='Pará', PB='Paraíba', PR='Paraná', PE='Pernambuco', PI='Piauí',
                  RJ='Rio de Janeiro', RN='Rio Grande do Norte', RS='Rio Grande do Sul', RO='Rondônia', RR='Roraima',
                  SC='Santa Catarina', SP='São Paulo', SE='Sergipe', TO='Tocantins')
REGION_STATES = {'North': 'AC AP AM PA RO RR TO', 'Northeast': 'AL BA CE MA PB PE PI RN SE',
                 'Central-West': 'DF GO MT MS', 'Southeast': 'ES MG RJ SP', 'South': 'PR RS SC'}
REGION = {s: r for r, ss in REGION_STATES.items() for s in ss.split()}
REGION_VN = {'North': 'Miền Bắc', 'Northeast': 'Đông Bắc', 'Central-West': 'Trung Tây', 'Southeast': 'Đông Nam', 'South': 'Miền Nam'}
BBOX = dict(lat=(-33.75, 5.28), lng=(-73.99, -34.79))     # hộp bao lãnh thổ Brazil

# ---------------------------------------------------------------- ngành hàng
TYPO = {'fashio_female_clothing': 'fashion_female_clothing', 'costruction_tools_garden': 'construction_tools_garden',
        'costruction_tools_tools': 'construction_tools_tools', 'home_confort': 'home_comfort'}
EXTRA_TRANSLATION = {'pc_gamer': 'pc_gamer', 'portateis_cozinha_e_preparadores_de_alimentos': 'portable_kitchen_food_preparers'}
CAT_GROUPS = {
    'Electronics & Computing': 'audio cine_photo computers computers_accessories consoles_games electronics fixed_telephony telephony tablets_printing_image pc_gamer',
    'Home & Living': 'bed_bath_table furniture_bedroom furniture_decor furniture_living_room furniture_mattress_and_upholstery home_appliances home_appliances_2 home_comfort home_comfort_2 housewares kitchen_dining_laundry_garden_furniture la_cuisine small_appliances small_appliances_home_oven_and_coffee air_conditioning office_furniture portable_kitchen_food_preparers',
    'Construction & Tools': 'construction_tools_construction construction_tools_lights construction_tools_safety construction_tools_garden construction_tools_tools garden_tools home_construction signaling_and_security',
    'Industry & Services': 'agro_industry_and_commerce industry_commerce_and_business market_place security_and_services',
    'Fashion & Accessories': 'fashion_female_clothing fashion_bags_accessories fashion_childrens_clothes fashion_male_clothing fashion_shoes fashion_sport fashion_underwear_beach luggage_accessories watches_gifts',
    'Health & Beauty': 'health_beauty perfumery diapers_and_hygiene',
    'Baby & Toys': 'baby toys',
    'Sports & Leisure': 'sports_leisure musical_instruments cool_stuff party_supplies christmas_supplies art arts_and_craftmanship pet_shop flowers',
    'Books & Media': 'books_general_interest books_imported books_technical cds_dvds_musicals dvds_blu_ray music stationery',
    'Food & Drink': 'food food_drink drinks',
    'Auto': 'auto', 'Unknown': 'unknown'}
CAT_GROUP = {c: g for g, cs in CAT_GROUPS.items() for c in cs.split()}
CAT_GROUP_VN = {'Electronics & Computing': 'Điện tử & Máy tính', 'Home & Living': 'Nhà cửa & Đời sống',
                'Construction & Tools': 'Xây dựng & Dụng cụ', 'Industry & Services': 'Công nghiệp & Dịch vụ',
                'Fashion & Accessories': 'Thời trang & Phụ kiện', 'Health & Beauty': 'Sức khỏe & Làm đẹp',
                'Baby & Toys': 'Mẹ bé & Đồ chơi', 'Sports & Leisure': 'Thể thao & Giải trí', 'Books & Media': 'Sách & Truyền thông',
                'Food & Drink': 'Thực phẩm & Đồ uống', 'Auto': 'Ô tô – Xe máy', 'Unknown': 'Chưa phân loại'}
CAT_VN = {
    'health_beauty': 'Sức khỏe & làm đẹp', 'computers_accessories': 'Phụ kiện máy tính', 'auto': 'Ô tô – xe máy',
    'bed_bath_table': 'Chăn ga, nhà tắm & bàn ăn', 'furniture_decor': 'Nội thất trang trí', 'sports_leisure': 'Thể thao & giải trí',
    'perfumery': 'Nước hoa', 'housewares': 'Đồ gia dụng', 'telephony': 'Điện thoại di động', 'watches_gifts': 'Đồng hồ & quà tặng',
    'food_drink': 'Thực phẩm & đồ uống', 'baby': 'Mẹ & bé', 'stationery': 'Văn phòng phẩm',
    'tablets_printing_image': 'Máy tính bảng, in ấn & ảnh', 'toys': 'Đồ chơi', 'fixed_telephony': 'Điện thoại bàn',
    'garden_tools': 'Dụng cụ làm vườn', 'fashion_bags_accessories': 'Túi xách & phụ kiện', 'small_appliances': 'Thiết bị điện nhỏ',
    'consoles_games': 'Máy chơi game', 'audio': 'Thiết bị âm thanh', 'fashion_shoes': 'Giày dép', 'cool_stuff': 'Đồ độc lạ',
    'luggage_accessories': 'Vali & phụ kiện du lịch', 'air_conditioning': 'Điều hòa không khí',
    'construction_tools_construction': 'Dụng cụ xây dựng', 'kitchen_dining_laundry_garden_furniture': 'Nội thất bếp, giặt & sân vườn',
    'construction_tools_garden': 'Dụng cụ xây dựng – sân vườn', 'fashion_male_clothing': 'Thời trang nam', 'pet_shop': 'Đồ thú cưng',
    'office_furniture': 'Nội thất văn phòng', 'market_place': 'Hàng sàn tổng hợp', 'electronics': 'Đồ điện tử',
    'home_appliances': 'Đồ điện gia dụng', 'party_supplies': 'Đồ dùng tiệc', 'home_comfort': 'Tiện nghi gia đình',
    'construction_tools_tools': 'Dụng cụ cầm tay', 'agro_industry_and_commerce': 'Nông nghiệp & thương mại',
    'furniture_mattress_and_upholstery': 'Nệm & bọc nội thất', 'books_technical': 'Sách kỹ thuật', 'home_construction': 'Vật liệu xây nhà',
    'musical_instruments': 'Nhạc cụ', 'furniture_living_room': 'Nội thất phòng khách', 'construction_tools_lights': 'Đèn chiếu sáng',
    'industry_commerce_and_business': 'Công nghiệp & kinh doanh', 'food': 'Thực phẩm', 'art': 'Nghệ thuật',
    'furniture_bedroom': 'Nội thất phòng ngủ', 'books_general_interest': 'Sách phổ thông', 'construction_tools_safety': 'Thiết bị bảo hộ',
    'fashion_underwear_beach': 'Đồ lót & đồ bơi', 'fashion_sport': 'Thời trang thể thao', 'signaling_and_security': 'Biển báo & an ninh',
    'computers': 'Máy tính', 'christmas_supplies': 'Đồ Giáng sinh', 'fashion_female_clothing': 'Thời trang nữ',
    'home_appliances_2': 'Đồ điện gia dụng (2)', 'books_imported': 'Sách nhập khẩu', 'drinks': 'Đồ uống', 'cine_photo': 'Máy ảnh & quay phim',
    'la_cuisine': 'Dụng cụ nấu ăn cao cấp', 'music': 'Âm nhạc', 'home_comfort_2': 'Tiện nghi gia đình (2)',
    'small_appliances_home_oven_and_coffee': 'Lò nướng & máy pha cà phê', 'cds_dvds_musicals': 'CD, DVD ca nhạc', 'dvds_blu_ray': 'DVD & Blu-ray',
    'flowers': 'Hoa', 'arts_and_craftmanship': 'Thủ công mỹ nghệ', 'diapers_and_hygiene': 'Tã & vệ sinh',
    'fashion_childrens_clothes': 'Thời trang trẻ em', 'security_and_services': 'An ninh & dịch vụ', 'pc_gamer': 'Máy tính chơi game',
    'portable_kitchen_food_preparers': 'Máy chế biến thực phẩm', 'unknown': 'Chưa phân loại'}

# ---------------------------------------------------------------- danh mục nhỏ
PAY_VN = {'credit_card': 'Thẻ tín dụng', 'boleto': 'Boleto (phiếu thanh toán)', 'voucher': 'Voucher', 'debit_card': 'Thẻ ghi nợ',
          'not_defined': 'Không xác định'}
STATUS_VN = {'delivered': 'Đã giao', 'shipped': 'Đang vận chuyển', 'canceled': 'Đã hủy', 'unavailable': 'Hết hàng / không khả dụng',
             'invoiced': 'Đã xuất hóa đơn', 'processing': 'Đang xử lý', 'created': 'Mới tạo', 'approved': 'Đã duyệt'}
STATUS_GROUP = {'delivered': 'Hoàn tất', 'shipped': 'Đang xử lý', 'invoiced': 'Đang xử lý', 'processing': 'Đang xử lý',
                'created': 'Đang xử lý', 'approved': 'Đang xử lý', 'canceled': 'Không thành công', 'unavailable': 'Không thành công'}
SCORE_VN = {1: 'Rất tệ', 2: 'Tệ', 3: 'Bình thường', 4: 'Tốt', 5: 'Rất tốt'}
SENTIMENT = {1: 'Tiêu cực (1–2)', 2: 'Tiêu cực (1–2)', 3: 'Trung lập (3)', 4: 'Tích cực (4–5)', 5: 'Tích cực (4–5)'}
WEEKDAY_VN = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ nhật']

# ---------------------------------------------------------------- mô tả cột nguồn (vai trò trong kho dữ liệu)
M = lambda role, tgt, desc: (role, tgt, desc)
FIELD_META = {
    'customers.customer_id': M('Khóa ngoại (FK)', 'fact_* (qua customer_key)', 'Mã khách gắn với MỖI đơn hàng (khóa nối orders ↔ customers)'),
    'customers.customer_unique_id': M('Khóa nghiệp vụ của chiều', 'dim_customer', 'Mã khách hàng thật – 1 khách có thể có nhiều customer_id'),
    'customers.customer_zip_code_prefix': M('Thuộc tính chiều', 'dim_geography', '5 chữ số đầu mã bưu chính của khách'),
    'customers.customer_city': M('Thuộc tính chiều', 'dim_geography', 'Thành phố của khách'),
    'customers.customer_state': M('Thuộc tính chiều', 'dim_geography', 'Bang (UF) của khách – 27 giá trị'),
    'geolocation.geolocation_zip_code_prefix': M('Khóa nghiệp vụ của chiều', 'dim_geography', '5 chữ số đầu mã bưu chính (nhiều dòng / zip)'),
    'geolocation.geolocation_lat': M('Thuộc tính chiều', 'dim_geography', 'Vĩ độ (có điểm nằm ngoài Brazil cần loại)'),
    'geolocation.geolocation_lng': M('Thuộc tính chiều', 'dim_geography', 'Kinh độ (có điểm nằm ngoài Brazil cần loại)'),
    'geolocation.geolocation_city': M('Thuộc tính chiều', 'dim_geography', 'Tên thành phố (nhiều biến thể có dấu / không dấu)'),
    'geolocation.geolocation_state': M('Thuộc tính chiều', 'dim_geography', 'Bang (UF)'),
    'orders.order_id': M('Chiều suy biến (degenerate)', 'fact_order / fact_sales', 'Mã đơn hàng – khóa chính của orders'),
    'orders.customer_id': M('Khóa ngoại (FK)', 'dim_customer', 'Khách đặt đơn'),
    'orders.order_status': M('Thuộc tính chiều', 'dim_order_status', 'Trạng thái đơn (8 giá trị)'),
    'orders.order_purchase_timestamp': M('Thời gian', 'dim_date', 'Thời điểm khách đặt hàng'),
    'orders.order_approved_at': M('Thời gian', 'dim_date', 'Thời điểm duyệt thanh toán'),
    'orders.order_delivered_carrier_date': M('Thời gian', 'dim_date', 'Thời điểm giao cho hãng vận chuyển'),
    'orders.order_delivered_customer_date': M('Thời gian', 'dim_date', 'Thời điểm giao tới khách'),
    'orders.order_estimated_delivery_date': M('Thời gian', 'dim_date', 'Ngày giao dự kiến cam kết với khách'),
    'items.order_id': M('Khóa ngoại (FK)', 'fact_sales', 'Đơn hàng chứa dòng sản phẩm'),
    'items.order_item_id': M('Chiều suy biến (degenerate)', 'fact_sales', 'Số thứ tự dòng trong đơn (không phải số lượng)'),
    'items.product_id': M('Khóa ngoại (FK)', 'dim_product', 'Sản phẩm'),
    'items.seller_id': M('Khóa ngoại (FK)', 'dim_seller', 'Người bán'),
    'items.shipping_limit_date': M('Thời gian', 'dim_date', 'Hạn chót người bán giao cho hãng vận chuyển'),
    'items.price': M('Số đo (measure)', 'fact_sales', 'Giá bán của dòng sản phẩm (BRL)'),
    'items.freight_value': M('Số đo (measure)', 'fact_sales', 'Phí vận chuyển của dòng sản phẩm (BRL)'),
    'payments.order_id': M('Khóa ngoại (FK)', 'fact_payment', 'Đơn hàng được thanh toán'),
    'payments.payment_sequential': M('Chiều suy biến (degenerate)', 'fact_payment', 'Thứ tự lần thanh toán trong đơn'),
    'payments.payment_type': M('Thuộc tính chiều', 'dim_payment_type', 'Phương thức thanh toán (5 giá trị, có "not_defined")'),
    'payments.payment_installments': M('Số đo (measure)', 'fact_payment', 'Số kỳ trả góp'),
    'payments.payment_value': M('Số đo (measure)', 'fact_payment', 'Giá trị thanh toán (BRL)'),
    'reviews.review_id': M('Chiều suy biến (degenerate)', 'fact_review', 'Mã đánh giá (không duy nhất toàn bảng)'),
    'reviews.order_id': M('Khóa ngoại (FK)', 'fact_review', 'Đơn hàng được đánh giá'),
    'reviews.review_score': M('Số đo (measure)', 'fact_review', 'Điểm đánh giá 1–5'),
    'reviews.review_comment_title': M('Văn bản tự do', '(không đưa vào DW)', 'Tiêu đề bình luận (phần lớn trống)'),
    'reviews.review_comment_message': M('Văn bản tự do', 'fact_review (has_comment, comment_length)', 'Nội dung bình luận'),
    'reviews.review_creation_date': M('Thời gian', 'dim_date', 'Ngày gửi khảo sát đánh giá'),
    'reviews.review_answer_timestamp': M('Thời gian', 'dim_date', 'Thời điểm khách trả lời khảo sát'),
    'products.product_id': M('Khóa nghiệp vụ của chiều', 'dim_product', 'Mã sản phẩm'),
    'products.product_category_name': M('Thuộc tính chiều', 'dim_product', 'Tên ngành hàng (tiếng Bồ Đào Nha)'),
    'products.product_name_lenght': M('Thuộc tính chiều', 'dim_product', 'Số ký tự tên sản phẩm (tên cột sai chính tả)'),
    'products.product_description_lenght': M('Thuộc tính chiều', 'dim_product', 'Số ký tự mô tả sản phẩm'),
    'products.product_photos_qty': M('Thuộc tính chiều', 'dim_product', 'Số ảnh sản phẩm'),
    'products.product_weight_g': M('Thuộc tính chiều', 'dim_product', 'Cân nặng (gram)'),
    'products.product_length_cm': M('Thuộc tính chiều', 'dim_product', 'Chiều dài đóng gói (cm)'),
    'products.product_height_cm': M('Thuộc tính chiều', 'dim_product', 'Chiều cao đóng gói (cm)'),
    'products.product_width_cm': M('Thuộc tính chiều', 'dim_product', 'Chiều rộng đóng gói (cm)'),
    'sellers.seller_id': M('Khóa nghiệp vụ của chiều', 'dim_seller', 'Mã người bán'),
    'sellers.seller_zip_code_prefix': M('Thuộc tính chiều', 'dim_geography', '5 chữ số đầu mã bưu chính của người bán'),
    'sellers.seller_city': M('Thuộc tính chiều', 'dim_seller', 'Thành phố người bán'),
    'sellers.seller_state': M('Thuộc tính chiều', 'dim_seller', 'Bang người bán'),
    'translation.product_category_name': M('Khóa nghiệp vụ của chiều', 'dim_product', 'Tên ngành hàng tiếng Bồ Đào Nha'),
    'translation.product_category_name_english': M('Thuộc tính chiều', 'dim_product', 'Tên ngành hàng tiếng Anh'),
}


# ---------------------------------------------------------------- hàm tiện ích
def unaccent(s):
    """Bỏ dấu (NFKD), chữ thường, gộp khoảng trắng: 'São  Paulo ' -> 'sao paulo'."""
    s = s.fillna('').map(lambda x: unicodedata.normalize('NFKD', x))
    s = s.str.replace(r'[̀-ͯ]', '', regex=True).str.replace('´', ' ', regex=False)
    return s.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def norm_text(s):
    """Khóa so khớp / từ cho văn bản: bỏ dấu, chữ thường, mọi ký tự không phải chữ-số thành khoảng trắng.
    "Alta Floresta D'Oeste" -> 'alta floresta doeste', "Sant'Ana" -> 'santana' (tên thành phố IBGE và bình luận đánh giá – N6, N7)."""
    s = s.fillna('').str.replace(r"['’`´]", '', regex=True)
    return unaccent(s).str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()


# Từ dừng tiếng Bồ Đào Nha (đã bỏ dấu) – loại khi tách từ bình luận. Giữ lại 'nao' vì phủ định mang nghĩa ("não recebi").
STOP_PT = set('''a o as os e de da do das dos em no na nos nas um uma uns umas para pra pro por com sem que se ao aos
    e foi era ser ter tem tinha tive teve meu minha meus minhas seu sua seus suas eu voce voces vc ele ela eles elas isso esse essa
    este esta aquele aquela isto aqui ali la ja ainda mais muito muita bem mas ou nem como quando onde porque pois ate entao so
    tambem sobre entre apos antes depois mesmo mesma tudo todo toda todos todas nada algo outro outra me te lhe nos lo los las
    num numa pelo pela pelos pelas estou estava estao sao fui vai vou ha seja sido sendo fazer feito fiz q pq'''.split())


def haversine(lat1, lng1, lat2, lng2):
    """Khoảng cách đường chim bay (km) giữa 2 tọa độ."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lng2 - lng1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def iqr_outliers(s):
    """Số giá trị nằm ngoài [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]."""
    s = s.dropna()
    q1, q3 = s.quantile([.25, .75])
    return int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())


def skew_label(sk):
    a = abs(sk)
    return 'Gần đối xứng' if a < .5 else ('Lệch vừa' if a < 1 else 'Lệch mạnh') + (' phải' if sk > 0 else ' trái')


def strength(r):
    a = abs(r)
    return 'Rất yếu' if a < .1 else 'Yếu' if a < .3 else 'Trung bình' if a < .5 else 'Mạnh' if a < .7 else 'Rất mạnh'


def vn(v, nd=0):
    """Định dạng số kiểu Việt Nam: 1234567.8 -> '1.234.567,8'."""
    s = f'{v:,.{nd}f}'
    return s.replace(',', '§').replace('.', ',').replace('§', '.')
