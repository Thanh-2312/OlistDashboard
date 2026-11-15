# Olist BI Dashboard

Bộ 8 dashboard (mã hóa theo Bài 3) cho dữ liệu Olist Brazilian E-commerce, đồ án môn Kho dữ liệu & BI.

Xem trực tuyến: https://thanh-2312.github.io/OlistDashboard/

- `index.html`: giao diện (Chart.js), cùng tông màu với dashboard Power BI / Excel.
- `data.js`: dữ liệu từng dòng (nén dạng cột) sinh bởi `_build/web_data.py` từ kho dữ liệu; trình duyệt tự lọc và tính measure giống Power BI (slicer + lọc chéo).

## Phần mở rộng: Lakehouse & AI

Xem trực tuyến: https://thanh-2312.github.io/OlistDashboard/mo_rong.html

- `mo_rong.html`: 7 trang gồm Lakehouse (API, JSON lồng nhau, văn bản), AI (thử đọc cảm xúc ngay trên trang, chủ đề phàn nàn, dự báo giao trễ) và mã nguồn.
- `mo_rong_data.js`: số liệu kết quả và mô hình cảm xúc (TF-IDF + hồi quy logistic) dạng JSON, sinh bởi `mo_rong/src/web_mo_rong.py`.
- `mo_rong/`: notebook N6, N7, mã nguồn, biểu đồ, mẫu dữ liệu – xem [mo_rong/README.md](mo_rong/README.md).
