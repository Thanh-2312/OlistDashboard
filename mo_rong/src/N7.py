# %% [markdown]
# # N7 – ỨNG DỤNG AI TRONG KHAI THÁC DỮ LIỆU: ĐỌC HIỂU BÌNH LUẬN VÀ DỰ BÁO ĐƠN GIAO TRỄ
#
# Dashboard (N5, Power BI) trả lời **"đã xảy ra chuyện gì"**. Notebook này dùng học máy trên chính dữ liệu của lakehouse để trả lời thêm
# **"khách đang nói gì"** (đọc 42 nghìn bình luận tự do – dữ liệu không cấu trúc từ N6) và **"đơn nào sắp trễ"** (dự báo trước khi xảy ra).
# Kết quả AI được **ghi ngược vào vùng Gold** để Power BI dùng như mọi bảng khác.
#
# | # | Câu hỏi nghiệp vụ | Loại AI | Dữ liệu (lakehouse) | Đầu ra | Ai dùng |
# |---|---|---|---|---|---|
# | 1 | Bình luận này **khen hay chê**? | Học có giám sát – phân loại văn bản (TF-IDF + hồi quy logistic) | `silver/review_text`, nhãn = số sao | Xác suất tiêu cực của từng bình luận; danh sách "khen sao nhưng chê lời" | Chăm sóc khách hàng |
# | 2 | Khách **phàn nàn về điều gì**? | Học không giám sát – mô hình chủ đề (NMF) | Bình luận 1–2 sao | 6 nhóm vấn đề, xu hướng theo tháng, liên hệ với giao trễ | Vận chuyển, Bán hàng |
# | 3 | **Đơn nào sắp giao trễ**? | Học có giám sát – cây tăng cường (gradient boosting) | `gold/fact_order`, `fact_sales`, `dim_*` | Điểm rủi ro trễ của mỗi đơn **lúc bàn giao cho hãng vận chuyển** | Vận chuyển (VC-V610) |
#
# **Nguyên tắc đánh giá trung thực**: huấn luyện trên **quá khứ**, kiểm tra trên **tương lai** (chia theo thời gian, không chia ngẫu nhiên)
# và luôn so với một cách làm đơn giản (baseline) để biết AI thêm được bao nhiêu.
#
# **Vì sao dùng mô hình học máy chạy trên máy thay vì gọi LLM (ChatGPT / Claude / Copilot)?** Không gửi bình luận của khách ra ngoài,
# không tốn phí, chạy lại ra đúng kết quả cũ, và giải thích được bằng trọng số của từng từ. LLM là hướng phát triển tiếp theo
# (hỏi đáp dữ liệu bằng ngôn ngữ tự nhiên) – nêu ở phần cuối.
#
# **Chạy sau N6.**

# %%
import time, warnings
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deltalake import DeltaTable, write_deltalake
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from olist_utils import *

warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 110)
LAKE = DATA / 'lakehouse'
CHART = PROJECT / 'assets' / 'charts'
CHART.mkdir(parents=True, exist_ok=True)
if not (LAKE / 'silver' / 'review_text').exists():
    raise RuntimeError('Chưa có silver/review_text – hãy chạy N6 trước')
MODEL_VERSION = 'ai-' + pd.Timestamp.now().strftime('%Y%m%d')
BLUE, ORANGE, GREEN, RED, GREY = '#1F5FA8', '#E07B1A', '#2E9E5B', '#C0392B', '#8C8C8C'
get = lambda layer, name: DeltaTable(str(LAKE / layer / name)).to_pandas()


def scores(y, p, thr=0.5):
    """Các chỉ số cho lớp dương (tiêu cực / giao trễ)."""
    yhat = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat),
                f1=f1_score(y, yhat), roc_auc=roc_auc_score(y, p) if len(set(p)) > 1 else np.nan)


# %% [markdown]
# ## 1. AI ĐỌC CẢM XÚC BÌNH LUẬN (phân loại văn bản)
# * **Nhãn có sẵn**: 1–2 sao = *tiêu cực*, 4–5 sao = *tích cực* (bỏ 3 sao vì trung lập, mơ hồ).
# * **Biến chữ thành số – TF-IDF**: mỗi bình luận thành một vector; từ / cụm 2 từ hay ký tự xuất hiện nhiều trong bình luận nhưng hiếm trong toàn bộ dữ liệu được trọng số cao.
#   Thêm n-gram **ký tự** (3–5 ký tự) để chịu được lỗi chính tả, thiếu dấu ("pessimo", "péssimo", "pesimo").
# * **Hồi quy logistic**: học trọng số cho từng từ → cộng lại thành xác suất "tiêu cực". Trọng số đọc được → biết vì sao mô hình kết luận.
# * **Chia theo thời gian**: học trên bình luận trước 01/04/2018, kiểm tra trên bình luận từ 01/04/2018 (mô hình chưa từng thấy).

# %%
rt = get('silver', 'review_text')
lab = rt[rt.review_score != 3].copy()
lab['y'] = (lab.review_score <= 2).astype(int)
CUT = pd.Timestamp('2018-04-01')
tr, te = lab[lab.review_date < CUT], lab[lab.review_date >= CUT]
STOP = sorted(STOP_PT)
print(f'Huấn luyện: {len(tr):,} bình luận (tiêu cực {tr.y.mean():.1%}) · kiểm tra: {len(te):,} (tiêu cực {te.y.mean():.1%})')

NEG_WORDS = {'nao', 'ruim', 'pessimo', 'pessima', 'horrivel', 'defeito', 'quebrado', 'errado', 'atraso', 'demora', 'faltando'}
rule = te.text_norm.str.split().map(lambda w: int(bool(NEG_WORDS & set(w))))
word_lr = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words=STOP),
                        LogisticRegression(C=4, max_iter=3000, class_weight='balanced'))


def full_model():
    return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                    TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
                         LogisticRegression(C=4, max_iter=3000, class_weight='balanced'))


t0 = time.perf_counter()
word_lr.fit(tr.text_norm, tr.y)
sent = full_model().fit(tr.text_norm, tr.y)
p_word, p_full = word_lr.predict_proba(te.text_norm)[:, 1], sent.predict_proba(te.text_norm)[:, 1]
print(f'Huấn luyện 2 mô hình: {time.perf_counter() - t0:.1f}s')
sm = save(pd.DataFrame([dict(mo_hinh='Luôn đoán "tích cực" (đa số)', **scores(te.y, np.zeros(len(te)))),
                        dict(mo_hinh=f'Quy tắc: có 1 trong {len(NEG_WORDS)} từ tiêu cực', **scores(te.y, rule.values.astype(float))),
                        dict(mo_hinh='AI: TF-IDF từ + hồi quy logistic', **scores(te.y, p_word)),
                        dict(mo_hinh='AI: TF-IDF từ + ký tự + hồi quy logistic (chọn)', **scores(te.y, p_full))]).round(3), 'ai_sentiment_metrics')
display(sm)

# Thử nghiệm phụ: 3 lớp (thêm "trung lập" = 3 sao) – khó hơn nhiều vì 3 sao vừa khen vừa chê
y3 = lambda d: np.select([d.review_score <= 2, d.review_score == 3], [0, 1], 2)
tr3, te3 = rt[rt.review_date < CUT], rt[rt.review_date >= CUT]
m3 = full_model().fit(tr3.text_norm, y3(tr3))
print(f'Phân 3 lớp (tiêu cực / trung lập / tích cực): độ chính xác {accuracy_score(y3(te3), m3.predict(te3.text_norm)):.1%}, '
      f'F1 trung bình 3 lớp {f1_score(y3(te3), m3.predict(te3.text_norm), average="macro"):.2f}')

# %%
cm = confusion_matrix(te.y, (p_full >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.0%})'.replace(',', '.'), ha='center', va='center', fontsize=12,
                color='white' if i == j else '#222222', fontweight='bold')
ax.set_xticks([0, 1], ['AI đoán: tích cực', 'AI đoán: tiêu cực']); ax.set_yticks([0, 1], ['Thực tế: 4–5 sao', 'Thực tế: 1–2 sao'])
ax.set_title(f'Bình luận từ 04/2018 (mô hình chưa thấy)\nĐộ chính xác {sm.accuracy.iloc[-1]:.1%} · AUC {sm.roc_auc.iloc[-1]:.3f}'.replace('.', ','),
             fontsize=11, fontweight='bold')
fig.tight_layout(); fig.savefig(CHART / 'n7_sentiment_confusion.png'); plt.close(fig)

# %% [markdown]
# ### Mô hình học được những từ nào? (trọng số của mô hình TF-IDF từ – dễ đọc hơn mô hình ký tự)

# %%
GLOSS = {'nao': 'không', 'nao recomendo': 'không khuyên mua', 'pessima': 'tệ hại', 'pessimo': 'tệ hại', 'nao recebi': 'chưa nhận được',
         'aguardando': 'đang chờ', 'ruim': 'tệ', 'inferior': 'kém hơn', 'nao gostei': 'không thích', 'horrivel': 'kinh khủng',
         'nao chegou': 'chưa đến', 'quebrado': 'bị vỡ', 'contato': 'liên hệ', 'recebi apenas': 'chỉ nhận được', 'diferente': 'khác mô tả',
         'demora': 'chậm', 'errado': 'sai', 'faltando': 'thiếu', 'insatisfeita': 'không hài lòng', 'insatisfeito': 'không hài lòng',
         'atrasada': 'bị trễ', 'atrasado': 'bị trễ', 'pedi': 'tôi đã đặt', 'pior': 'tệ nhất', 'quero': 'tôi muốn', 'aguardo': 'tôi đang chờ',
         'comprei': 'tôi đã mua', 'defeito': 'lỗi', 'devolver': 'trả hàng', 'cancelar': 'hủy', 'parabens': 'xin chúc mừng / khen',
         'recomendo': 'khuyên mua', 'lindo': 'đẹp', 'linda': 'đẹp', 'rapido': 'nhanh', 'rapida': 'nhanh', 'perfeito': 'hoàn hảo',
         'amei': 'rất thích', 'adorei': 'rất thích', 'embalado': 'đóng gói kỹ', 'excelente': 'xuất sắc', 'otima': 'tuyệt vời',
         'otimo': 'tuyệt vời', 'obrigada': 'cảm ơn', 'obrigado': 'cảm ơn', 'produto bom': 'sản phẩm tốt', 'certo': 'đúng', 'bonito': 'đẹp',
         'entregue prazo': 'giao đúng hạn', 'sempre': 'luôn luôn', 'chegou prazo': 'đến đúng hạn', 'certinho': 'đúng y', 'satisfeita': 'hài lòng',
         'satisfeito': 'hài lòng', 'unica': 'duy nhất', 'previsto': 'dự kiến', 'direitinho': 'chu đáo', 'entrega correios': 'giao qua bưu điện',
         'bom': 'tốt', 'gostei': 'thích', 'antes': 'trước (sớm)', 'qualidade': 'chất lượng', 'super': 'rất', 'tudo certo': 'mọi thứ ổn',
         'muito bom': 'rất tốt', 'top': 'tuyệt', 'maravilhoso': 'tuyệt vời', 'antes prazo': 'trước hạn'}
vec, lr = word_lr.named_steps['tfidfvectorizer'], word_lr.named_steps['logisticregression']
terms, coef = np.array(vec.get_feature_names_out()), lr.coef_[0]
order = coef.argsort()
words = save(pd.concat([pd.DataFrame(dict(nhom='Đẩy về tiêu cực', tu=terms[order[::-1][:15]], trong_so=coef[order[::-1][:15]])),
                        pd.DataFrame(dict(nhom='Đẩy về tích cực', tu=terms[order[:15]], trong_so=coef[order[:15]]))])
             .assign(nghia=lambda d: d.tu.map(GLOSS).fillna('')), 'ai_sentiment_words')
display(words)

fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
for a, (grp, col) in zip(ax, [('Đẩy về tiêu cực', RED), ('Đẩy về tích cực', GREEN)]):
    w = words[words.nhom == grp].head(12).iloc[::-1]
    a.barh([f'{t}  ({g})' if g else t for t, g in zip(w.tu, w.nghia)], w.trong_so.abs(), color=col)
    a.set_title(f'12 từ {grp.lower()} mạnh nhất', fontsize=11, fontweight='bold')
    a.spines[['top', 'right']].set_visible(False); a.set_xlabel('Độ lớn trọng số')
fig.suptitle('Từ ngữ mô hình dùng để nhận ra bình luận chê / khen (tiếng Bồ Đào Nha, đã bỏ dấu)', fontsize=12, fontweight='bold')
fig.tight_layout(); fig.savefig(CHART / 'n7_sentiment_words.png'); plt.close(fig)

# %% [markdown]
# ### Ứng dụng: chấm điểm mọi bình luận và tìm chỗ **số sao và lời viết mâu thuẫn**
# * Bình luận có sao (1–2, 4–5): dùng **dự đoán chéo 5 phần** (cross-validation) – mỗi bình luận được chấm bởi mô hình *không học* chính nó.
# * Bình luận 3 sao: chấm bằng mô hình học trên toàn bộ dữ liệu có nhãn → biết 3 sao nghiêng về khen hay chê.
# * "Khen sao chê lời" (≥ 4 sao nhưng xác suất tiêu cực ≥ 80%): khách cho điểm cao nhưng vẫn có vấn đề → CSKH nên liên hệ.

# %%
t0 = time.perf_counter()
lab['p_negative'] = cross_val_predict(full_model(), lab.text_norm, lab.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
final_sent = full_model().fit(lab.text_norm, lab.y)
neu = rt[rt.review_score == 3].copy()
neu['p_negative'] = final_sent.predict_proba(neu.text_norm)[:, 1]
print(f'Chấm điểm {len(lab) + len(neu):,} bình luận: {time.perf_counter() - t0:.1f}s')
scored = pd.concat([lab.drop(columns='y'), neu], ignore_index=True)
scored['sentiment_ai'] = np.where(scored.p_negative >= 0.5, 'Tiêu cực', 'Tích cực')
scored['mismatch'] = np.select([(scored.review_score >= 4) & (scored.p_negative >= 0.8), (scored.review_score <= 2) & (scored.p_negative <= 0.2)],
                               ['Khen sao, chê lời', 'Chê sao, khen lời'], '')
mm = save(scored.groupby('review_score').agg(binh_luan=('review_id', 'size'), pct_ai_tieu_cuc=('p_negative', lambda p: (p >= 0.5).mean()),
                                              khen_sao_che_loi=('mismatch', lambda m: (m == 'Khen sao, chê lời').sum()),
                                              che_sao_khen_loi=('mismatch', lambda m: (m == 'Chê sao, khen lời').sum())).reset_index(), 'ai_sentiment_by_score')
display(mm.round(3))
ex = save(scored[scored.mismatch == 'Khen sao, chê lời'].sort_values('p_negative', ascending=False)[['review_score', 'p_negative', 'text_masked']].head(8),
          'ai_sentiment_mismatch_examples')
print('Ví dụ "khen sao, chê lời" (văn bản đã che dữ liệu cá nhân):')
display(ex.round(3))

# %% [markdown]
# ## 2. AI TÌM CHỦ ĐỀ PHÀN NÀN (mô hình chủ đề NMF – học không giám sát)
# Không ai gán nhãn "bình luận này nói về giao hàng hay chất lượng". **NMF** (phân rã ma trận không âm) tự tách ma trận
# *bình luận × từ* thành *bình luận × chủ đề* và *chủ đề × từ*: mỗi chủ đề là một nhóm từ hay đi cùng nhau.
# * Chạy trên **mọi bình luận 1–2 sao**, 8 chủ đề, bỏ thêm các từ chung của ngành (produto, loja, pedido...).
# * **Đặt tên chủ đề tự động**: mỗi nhóm vấn đề có vài từ khóa gợi ý; chủ đề được gán cho nhóm có từ khóa nằm cao nhất trong danh sách từ của nó.
#   Nhiều chủ đề có thể cùng một nhóm (ví dụ "não recebi", "não chegou", "consta entregue" đều là *chưa nhận được hàng*).

# %%
neg = scored[scored.review_score <= 2].copy()
DOMAIN = {'produto', 'produtos', 'loja', 'lannister', 'compra', 'pedido', 'comprei'}
tv = TfidfVectorizer(stop_words=sorted(STOP_PT | DOMAIN), min_df=5, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True)
Xn = tv.fit_transform(neg.text_norm)
nmf = NMF(n_components=8, random_state=42, init='nndsvda', max_iter=500).fit(Xn)
W, tterms = nmf.transform(Xn), np.array(tv.get_feature_names_out())
THEMES = {'Chưa nhận được hàng': ['recebi', 'chegou', 'entregue', 'consta'],
          'Giao chậm, quá hạn': ['prazo', 'demora', 'atraso', 'entrega', 'passou'],
          'Giao thiếu, còn chờ phần còn lại': ['apenas', 'dois', 'duas', 'unidades', 'aguardando', 'fiscal'],
          'Hàng lỗi, sai, hỏng': ['defeito', 'errado', 'quebrado', 'veio', 'troca', 'faltando'],
          'Chất lượng kém, không như mô tả': ['qualidade', 'pessima', 'ruim', 'recomendo', 'gostei', 'diferente'],
          'Đòi hoàn tiền, trả hàng': ['dinheiro', 'devolver', 'devolucao', 'volta', 'reembolso', 'estorno']}


def label_topic(top):
    """Điểm của mỗi nhóm = tổng (15 − thứ hạng) của các từ trong top 15 chứa từ khóa gợi ý của nhóm."""
    pts = {th: sum(15 - r for r, term in enumerate(top) if set(term.split()) & set(seeds)) for th, seeds in THEMES.items()}
    return max(pts, key=pts.get)


TOP = [list(tterms[c.argsort()[::-1][:15]]) for c in nmf.components_]
topics = pd.DataFrame(dict(chu_de=range(8), tu_dac_trung=[', '.join(t[:10]) for t in TOP], nhom_van_de=[label_topic(t) for t in TOP]))
assert set(topics.nhom_van_de) == set(THEMES), 'có nhóm vấn đề không tìm thấy chủ đề nào'
neg['theme'] = topics.nhom_van_de.values[W.argmax(1)]
neg.loc[W.max(1) == 0, 'theme'] = 'Khác (quá ngắn)'
topics['so_binh_luan'] = [int((W.argmax(1) == i).sum()) for i in range(8)]
display(save(topics, 'ai_topics'))

# %%
fo = get('gold', 'fact_order')[['order_id', 'order_key', 'delay_days', 'is_late', 'is_delivered', 'delivery_status']]
neg = neg.merge(fo, on='order_id', how='left')
th = save(neg.groupby('theme').agg(binh_luan=('review_id', 'size'), pct_don_tre=('is_late', 'mean'),
                                   pct_chua_giao=('is_delivered', lambda s: 1 - s.mean()), tre_tb_ngay=('delay_days', 'mean'))
          .assign(ty_trong=lambda d: d.binh_luan / d.binh_luan.sum()).sort_values('binh_luan', ascending=False).reset_index(), 'ai_themes')
display(th.round(3))
deliv = th[th.theme.isin(['Chưa nhận được hàng', 'Giao chậm, quá hạn', 'Giao thiếu, còn chờ phần còn lại'])].ty_trong.sum()
print(f'Ba nhóm liên quan khâu giao hàng chiếm {deliv:.0%} bình luận tiêu cực')

neg['ym'] = neg.review_date.dt.strftime('%Y-%m')
trend = save(pd.crosstab(neg.ym, neg.theme).loc['2017-01':'2018-08'], 'ai_theme_trend', index=True)
ORDER = [t for t in th.theme if t in THEMES]
COL = dict(zip(ORDER, [RED, ORANGE, '#F2B134', BLUE, '#7A5195', GREY]))
fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.5]))
t2 = th[th.theme.isin(ORDER)].iloc[::-1]
bars = ax[0].barh(t2.theme, t2.ty_trong * 100, color=[COL[x] for x in t2.theme])
ax[0].bar_label(bars, labels=[f'{s:.0%} · {l:.0%} đơn trễ' for s, l in zip(t2.ty_trong, t2.pct_don_tre)], fontsize=9, padding=3)
ax[0].set_xlim(0, t2.ty_trong.max() * 160); ax[0].set_xticks([])
ax[0].set_title('Nhóm vấn đề trong bình luận 1–2 sao\n(tỷ trọng · % đơn của nhóm bị giao trễ)', fontsize=11, fontweight='bold')
ax[1].stackplot(range(len(trend)), *[trend[c] for c in ORDER], labels=ORDER, colors=[COL[c] for c in ORDER], alpha=0.9)
ax[1].set_xticks(range(len(trend)), trend.index, rotation=60, fontsize=8)
ax[1].set_title('Số bình luận tiêu cực theo tháng và nhóm vấn đề', fontsize=11, fontweight='bold')
ax[1].legend(fontsize=8, frameon=False, loc='upper left')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(CHART / 'n7_complaint_themes.png'); plt.close(fig)
peak = trend.sum(1).idxmax()
print(f'Tháng nhiều bình luận tiêu cực nhất: {peak} ({int(trend.loc[peak].sum())} bình luận; nhóm lớn nhất: {trend.loc[peak].idxmax()})')

# %% [markdown]
# ## 3. AI DỰ BÁO ĐƠN SẮP GIAO TRỄ (học có giám sát trên dữ liệu có cấu trúc)
# **Thời điểm dự báo**: khi người bán bàn giao kiện hàng cho hãng vận chuyển – lúc đó vẫn còn thời gian để can thiệp
# (ưu tiên tuyến, đổi hãng, báo trước cho khách). Chỉ dùng thông tin **đã biết tại thời điểm đó** (không dùng ngày giao thực tế → tránh "nhìn trước tương lai").
#
# | Đặc trưng | Ý nghĩa |
# |---|---|
# | `days_left` | Số ngày còn lại đến hạn hẹn giao, tính từ lúc bàn giao cho hãng vận chuyển |
# | `handling_days`, `approve_hours` | Người bán xử lý mất bao lâu; duyệt thanh toán mất bao lâu |
# | `estimated_days` | Số ngày hẹn giao lúc đặt hàng |
# | `distance_km`, `same_state` | Khoảng cách người bán → khách; cùng bang hay không |
# | `weight_g`, `volume_cm3`, `n_items`, `n_sellers`, `freight_total`, `order_value` | Kiện hàng to / nặng / nhiều người bán hay không |
# | `customer_state`, `seller_state`, `category_group` | Tuyến và loại hàng |
# | `purchase_month`, `carrier_weekday`, `installments` | Mùa vụ, ngày trong tuần, cách trả tiền |
#
# Mô hình **HistGradientBoosting** (hàng trăm cây quyết định nhỏ, cây sau sửa lỗi cây trước). Học trên đơn đặt **năm 2017**, kiểm tra trên đơn **01–08/2018**.

# %%
so = get('silver', 'orders')[['order_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_carrier_date', 'order_estimated_delivery_date']]
fod = get('gold', 'fact_order')
fs = get('gold', 'fact_sales')[['order_id', 'order_item_id', 'seller_key', 'product_key', 'distance_km']]
dp = get('gold', 'dim_product')[['product_key', 'weight_g', 'volume_cm3', 'category_group_vn']]
dsel = get('gold', 'dim_seller')[['seller_key', 'state']].rename(columns={'state': 'seller_state'})
dgeo = get('gold', 'dim_geography')[['geo_key', 'state']].rename(columns={'geo_key': 'customer_geo_key', 'state': 'customer_state'})
it = fs.merge(dp, on='product_key', how='left').merge(dsel, on='seller_key', how='left').sort_values(['order_id', 'order_item_id'])
agg = it.groupby('order_id').agg(distance_km=('distance_km', 'max'), weight_g=('weight_g', 'sum'), volume_cm3=('volume_cm3', 'sum'),
                                  seller_state=('seller_state', 'first'), category_group=('category_group_vn', 'first'))
d = fod.merge(so, on='order_id').merge(agg, on='order_id', how='inner').merge(dgeo, on='customer_geo_key', how='left')
d = d[d.order_delivered_carrier_date.notna() & (d.dq_flag_timeline == 0)].copy()
d['days_left'] = (d.order_estimated_delivery_date - d.order_delivered_carrier_date).dt.total_seconds() / 86400
d['purchase_month'] = d.order_purchase_timestamp.dt.month
d['carrier_weekday'] = d.order_delivered_carrier_date.dt.dayofweek
d['same_state'] = (d.customer_state == d.seller_state).astype(int)
d['installments'] = d.payment_installments
FEAT = ['days_left', 'handling_days', 'approve_hours', 'estimated_days', 'distance_km', 'same_state', 'weight_g', 'volume_cm3', 'n_items', 'n_sellers',
        'freight_total', 'order_value', 'customer_state', 'seller_state', 'category_group', 'purchase_month', 'carrier_weekday', 'installments']
CAT = ['customer_state', 'seller_state', 'category_group']
for c in FEAT:
    d[c] = d[c].astype('category') if c in CAT else pd.to_numeric(d[c], errors='coerce').astype(float)
labeled = d[(d.is_delivered == 1) & d.is_late.notna() & (d.order_purchase_timestamp < '2018-09-01')].copy()
labeled['is_late'] = labeled.is_late.astype(int)
dtr = labeled[labeled.order_purchase_timestamp < '2018-01-01']
dte = labeled[labeled.order_purchase_timestamp >= '2018-01-01']
transit = d[d.order_status == 'shipped'].copy()                  # đang trên đường giao: chưa biết kết quả → cần dự báo
print(f'Học: {len(dtr):,} đơn (trễ {dtr.is_late.mean():.1%}) · kiểm tra: {len(dte):,} đơn (trễ {dte.is_late.mean():.1%}) · '
      f'đang vận chuyển: {len(transit):,} đơn')


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, categorical_features='from_dtype', random_state=0)


t0 = time.perf_counter()
late_m = gbm().fit(dtr[FEAT], dtr.is_late)
p_te = late_m.predict_proba(dte[FEAT])[:, 1]
print(f'Huấn luyện: {time.perf_counter() - t0:.1f}s')
base = dte.is_late.mean()
lm = save(pd.DataFrame([dict(mo_hinh='Chọn ngẫu nhiên', roc_auc=0.5, pr_auc=base),
                        dict(mo_hinh='Quy tắc: còn ít ngày đến hạn hẹn thì rủi ro cao', roc_auc=roc_auc_score(dte.is_late, -dte.days_left),
                             pr_auc=average_precision_score(dte.is_late, -dte.days_left)),
                        dict(mo_hinh='AI: HistGradientBoosting (18 đặc trưng)', roc_auc=roc_auc_score(dte.is_late, p_te),
                             pr_auc=average_precision_score(dte.is_late, p_te))]).round(3), 'ai_late_metrics')
display(lm)

dte = dte.assign(p_late=p_te)
dte['decile'] = 10 - pd.qcut(dte.p_late.rank(method='first'), 10, labels=False)     # 1 = 10% đơn rủi ro cao nhất
lift = dte.groupby('decile').agg(don=('order_id', 'size'), don_tre=('is_late', 'sum'), ty_le_tre=('is_late', 'mean'), diem_rui_ro_min=('p_late', 'min'))
lift['pct_don_tre_bat_duoc'] = lift.don_tre / lift.don_tre.sum()
lift['luy_ke_bat_duoc'] = lift.pct_don_tre_bat_duoc.cumsum()
lift['lift'] = lift.ty_le_tre / base
lift = save(lift.reset_index(), 'ai_late_lift')
display(lift.round(3))
top = lift.iloc[0]
print(f'10% đơn rủi ro cao nhất: tỷ lệ trễ {top.ty_le_tre:.1%} (gấp {top.lift:.1f} lần mức chung {base:.1%}), bắt được {top.pct_don_tre_bat_duoc:.0%} '
      f'số đơn trễ; 30% đơn rủi ro cao nhất bắt được {lift.luy_ke_bat_duoc.iloc[2]:.0%}')

# %%
FEAT_VN = {'days_left': 'Số ngày còn lại đến hạn hẹn', 'handling_days': 'Số ngày người bán xử lý', 'approve_hours': 'Số giờ duyệt thanh toán',
           'estimated_days': 'Số ngày hẹn giao', 'distance_km': 'Khoảng cách người bán → khách', 'same_state': 'Cùng bang',
           'weight_g': 'Cân nặng kiện hàng', 'volume_cm3': 'Thể tích kiện hàng', 'n_items': 'Số sản phẩm', 'n_sellers': 'Số người bán',
           'freight_total': 'Phí vận chuyển', 'order_value': 'Giá trị đơn', 'customer_state': 'Bang của khách', 'seller_state': 'Bang người bán',
           'category_group': 'Nhóm ngành', 'purchase_month': 'Tháng đặt hàng', 'carrier_weekday': 'Thứ bàn giao vận chuyển', 'installments': 'Số kỳ trả góp'}
samp = dte.sample(min(20000, len(dte)), random_state=0)
t0 = time.perf_counter()
pi = permutation_importance(late_m, samp[FEAT], samp.is_late, scoring='roc_auc', n_repeats=5, random_state=0)
imp = save(pd.DataFrame(dict(dac_trung=FEAT, ten=[FEAT_VN[f] for f in FEAT], giam_auc=pi.importances_mean, do_lech=pi.importances_std))
           .sort_values('giam_auc', ascending=False), 'ai_late_importance')
print(f'Đo mức quan trọng (xáo trộn từng đặc trưng): {time.perf_counter() - t0:.1f}s')
display(imp.round(4))

fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=150, gridspec_kw=dict(width_ratios=[1.1, 1]))
bars = ax[0].bar(lift.decile.astype(str), lift.ty_le_tre * 100, color=[RED] + [ORANGE] * 2 + [BLUE] * 7)
ax[0].axhline(base * 100, color='#333333', ls='--', lw=1)
ax[0].text(9.4, base * 100 + 0.6, f'Mức chung {base:.1%}'.replace('.', ','), ha='right', fontsize=9)
ax[0].bar_label(bars, labels=[f'{x:.0%}' for x in lift.ty_le_tre], fontsize=8)
ax[0].set_xlabel('Nhóm 10% đơn theo điểm rủi ro (1 = rủi ro cao nhất)'); ax[0].set_ylabel('% đơn giao trễ thực tế')
ax[0].set_title(f'Đơn 01–08/2018: AUC {lm.roc_auc.iloc[-1]:.2f}'.replace('.', ',') + f' · nhóm 1 trễ gấp {top.lift:.1f} lần'.replace('.', ','),
                fontsize=11, fontweight='bold')
i2 = imp.head(10).iloc[::-1]
ax[1].barh(i2.ten, i2.giam_auc, color=BLUE, xerr=i2.do_lech, ecolor='#999999')
ax[1].set_title('10 đặc trưng quan trọng nhất\n(AUC giảm bao nhiêu khi xáo trộn đặc trưng đó)', fontsize=11, fontweight='bold')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(CHART / 'n7_late_risk.png'); plt.close(fig)

# %% [markdown]
# ## 4. GHI KẾT QUẢ AI NGƯỢC VÀO LAKEHOUSE (Gold → Publish cho Power BI)
# * `gold/review_ai`: xác suất tiêu cực, nhãn AI, nhóm vấn đề, cờ mâu thuẫn sao / lời của từng bình luận.
# * `gold/order_late_risk`: điểm rủi ro trễ của từng đơn (mô hình học lại trên toàn bộ đơn đã có kết quả), mức rủi ro theo ngưỡng của tập kiểm tra,
#   gồm cả **đơn đang vận chuyển** – danh sách cần theo dõi ngay.
# * Mỗi lần chạy ghi thêm một **phiên bản Delta** kèm `model_version` → so được kết quả giữa các phiên bản mô hình bằng du hành thời gian.

# %%
final_late = gbm().fit(labeled[FEAT], labeled.is_late)
q = dte.p_late.quantile([0.7, 0.9]).values                           # ngưỡng 30% và 10% rủi ro cao nhất của tập kiểm tra
d['risk_score'] = final_late.predict_proba(d[FEAT])[:, 1]
d['risk_band'] = np.select([d.risk_score >= q[1], d.risk_score >= q[0]], ['Rất cao (top 10%)', 'Cao (top 30%)'], 'Bình thường')
d['split'] = np.select([d.order_id.isin(dtr.order_id), d.order_id.isin(dte.order_id), d.order_status == 'shipped'],
                       ['Học (2017)', 'Kiểm tra (01–08/2018)', 'Đang vận chuyển'], 'Khác')
olr = d[['order_key', 'order_id', 'split', 'risk_score', 'risk_band', 'is_late']].assign(model_version=MODEL_VERSION)
fr = get('gold', 'fact_review')[['review_key', 'review_id', 'order_id']]
rai = scored.merge(neg[['review_id', 'order_id', 'theme']], on=['review_id', 'order_id'], how='left') \
            .merge(fr, on=['review_id', 'order_id'])[['review_key', 'review_score', 'p_negative', 'sentiment_ai', 'theme', 'mismatch']] \
            .assign(model_version=MODEL_VERSION)
assert len(rai) == len(scored)
for name, df in [('review_ai', rai), ('order_late_risk', olr)]:
    write_deltalake(str(LAKE / 'gold' / name), pa.Table.from_pandas(df, preserve_index=False), mode='overwrite', schema_mode='overwrite')
    pq.write_table(DeltaTable(str(LAKE / 'gold' / name)).to_pyarrow_table(), LAKE / 'publish' / f'{name}.parquet')
    print(f'gold/{name}: {len(df):,} dòng · phiên bản Delta {DeltaTable(str(LAKE / "gold" / name)).version()} · publish/{name}.parquet')
watch = save(olr[olr.split == 'Đang vận chuyển'].merge(d[['order_id', 'customer_state', 'seller_state', 'days_left']], on='order_id')
             .sort_values('risk_score', ascending=False).head(10), 'ai_transit_watchlist')
print('Đơn đang vận chuyển có rủi ro trễ cao nhất (danh sách cho bộ phận vận chuyển):')
display(watch.round(3))
display(olr.groupby(['split', 'risk_band']).size().unstack(fill_value=0))

# %% [markdown]
# ## 5. TỔNG KẾT: AI MANG LẠI GÌ, GIỚI HẠN Ở ĐÂU

# %%
SUM = [('Đọc cảm xúc bình luận', 'TF-IDF từ + ký tự, hồi quy logistic (có giám sát)',
        f"Bình luận chưa từng thấy (từ 04/2018): đúng {sm.accuracy.iloc[-1]:.1%}, AUC {sm.roc_auc.iloc[-1]:.3f}; quy tắc từ khóa chỉ đúng {sm.accuracy.iloc[1]:.1%}",
        f"Chấm tự động {len(scored):,} bình luận; {int((scored.mismatch == 'Khen sao, chê lời').sum())} bình luận khen sao nhưng chê lời → CSKH liên hệ"),
       ('Tìm chủ đề phàn nàn', 'NMF 8 chủ đề → 6 nhóm vấn đề (không giám sát)',
        f"{len(neg):,} bình luận 1–2 sao được xếp nhóm; nhóm liên quan giao hàng chiếm {deliv:.0%}",
        f"Đỉnh {peak}: nhóm \"{trend.loc[peak].idxmax()}\" tăng mạnh → khớp đỉnh % giao trễ trên dashboard"),
       ('Dự báo đơn sắp giao trễ', 'HistGradientBoosting, 18 đặc trưng lúc bàn giao vận chuyển (có giám sát)',
        f"Đơn 01–08/2018: AUC {lm.roc_auc.iloc[-1]:.2f} (quy tắc đơn giản {lm.roc_auc.iloc[1]:.2f}); nhóm 10% rủi ro nhất trễ {top.ty_le_tre:.0%}, gấp {top.lift:.1f} lần",
        f"Theo dõi 10% đơn rủi ro nhất là bắt được {top.pct_don_tre_bat_duoc:.0%} đơn sẽ trễ; {len(transit):,} đơn đang vận chuyển được chấm điểm")]
summ = save(pd.DataFrame(SUM, columns=['bai_toan', 'phuong_phap', 'ket_qua_do_duoc', 'gia_tri_nghiep_vu']), 'ai_summary')
display(summ)
LIMITS = ['Nhãn cảm xúc lấy từ số sao: có khách chấm sao không khớp lời viết → nhãn có nhiễu (chính là các ca "mâu thuẫn" ở mục 1).',
          'Mô hình văn bản chỉ học tiếng Bồ Đào Nha của bộ dữ liệu này; gặp từ lóng / sản phẩm mới cần học lại định kỳ.',
          'Năm 2018 tỷ lệ trễ cao hơn 2017 (dữ liệu thay đổi theo thời gian) nên mô hình trễ cần học lại hằng tháng; điểm rủi ro là dự báo, không phải nguyên nhân.',
          'Hướng tiếp: dùng LLM (Claude / Copilot) để tóm tắt bình luận và hỏi đáp dữ liệu bằng tiếng Việt trên các bảng Gold; cần API key và quy định che dữ liệu cá nhân.']
for x in LIMITS:
    print('•', x)
