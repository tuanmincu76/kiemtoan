# 🏦 Web App Phát hiện Giao dịch Bất thường (Fraud Detection Dashboard)

Ứng dụng web trực quan và tương tác được xây dựng trên nền tảng **Streamlit** giúp phát hiện các giao dịch gian lận hoặc tài khoản có hành vi bất thường. Ứng dụng kết hợp giữa phân tích bộ quy tắc nghiệp vụ (Rule-based) và Mô hình Học máy (Machine Learning) được huấn luyện trực tiếp trên dữ liệu.

---

## 🌟 Tính năng chính

1. **Phân tích giao dịch thô (Rule-based Analysis)**:
   - Tự động phát hiện các giao dịch bất thường bằng bộ quy tắc thông minh (Heuristics):
     - **STR (Large Transaction)**: Giao dịch có giá trị lớn vượt ngưỡng cấu hình.
     - **NGT (Night Transaction)**: Giao dịch lớn phát sinh vào ban đêm (22:00 - 05:00).
     - **EMP (Employee Anomaly)**: Nhân viên thực hiện giao dịch lớn bất thường.
     - **RND (Round Amount)**: Giao dịch có số tiền chẵn lớn bất thường.
     - **MIX (Mixed Suspicious)**: Các giao dịch có định dạng mã bất thường.
   - Dashboard trực quan hóa Plotly: Biểu đồ phân bố theo địa điểm, kênh giao dịch, loại giao dịch và biểu đồ xu hướng theo thời gian.
   - Hỗ trợ xem và xuất (Download) danh sách giao dịch bất thường dưới dạng CSV.

2. **Huấn luyện & So sánh Mô hình Học máy (Machine Learning)**:
   - Tái hiện chính xác các mô hình học máy từ notebook: **Logistic Regression**, **Decision Tree**, và **Random Forest**.
   - Hỗ trợ 3 nguồn dữ liệu huấn luyện:
     - Dữ liệu trích xuất tự động từ tệp giao dịch thô `transactions_Q1_demo.csv`.
     - Dữ liệu giả lập (Synthetic data) phỏng theo phân phối trong notebook.
     - Tải lên tệp huấn luyện tùy chỉnh (.csv).
   - So sánh trực quan các chỉ số hiệu năng (Accuracy, Precision, Recall, F1-Score).
   - Biểu đồ nhiệt Ma trận nhầm lẫn (Confusion Matrix Heatmap) và biểu đồ Mức độ quan trọng của đặc trưng (Feature Importance).

3. **Dự đoán rủi ro (Prediction Module)**:
   - **Dự đoán đơn lẻ**: Nhập trực tiếp 14 đặc trưng hành vi khách hàng bằng các thanh trượt trực quan.
   - **Dự đoán hàng loạt (Batch)**: Tải lên tệp Excel/CSV khách hàng mới cần kiểm tra, mô hình tự động dự báo và trả về danh sách kèm xác suất rủi ro để tải xuống.

---

## 🛠️ Cài đặt và Chạy cục bộ (Local Installation)

### Bước 1: Cài đặt Python
Đảm bảo bạn đã cài đặt Python (phiên bản 3.9 đến 3.11 được khuyến nghị). Nếu chưa, bạn có thể tải về từ [python.org](https://www.python.org/downloads/).

### Bước 2: Tải mã nguồn và cài đặt thư viện phụ thuộc
Mở Command Prompt/Terminal trong thư mục dự án và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng
Khởi chạy ứng dụng Streamlit bằng lệnh:
```bash
streamlit run app.py
```
Sau khi chạy, trình duyệt web sẽ tự động mở trang ứng dụng tại địa chỉ `http://localhost:8501`.

---

## 🚀 Hướng dẫn Triển khai lên Streamlit Community Cloud (Free)

Để ứng dụng của bạn chạy trực tuyến và chia sẻ được với người khác, bạn có thể triển khai miễn phí lên Streamlit Cloud:

1. **Đưa mã nguồn lên GitHub**:
   - Tạo một repository mới trên GitHub (ví dụ: `giao-dich-bat-thuong`).
   - Đẩy 4 tệp sau lên GitHub:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `transactions_Q1_demo.csv` (dữ liệu demo)
     *(Lưu ý: Không cần đẩy các thư mục ảo `.venv` hoặc tệp tạm khác).*

2. **Triển khai trên Streamlit Cloud**:
   - Truy cập trang web [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
   - Bấm nút **New app** (hoặc **Create app**).
   - Chọn Repository chứa dự án của bạn, chọn nhánh `main` (hoặc `master`), và nhập **Main file path** là `app.py`.
   - Bấm **Deploy!** 
   - Đợi vài phút để hệ thống cài đặt môi trường và khởi chạy. Ứng dụng của bạn sẽ hoạt động trực tuyến với một liên kết chia sẻ dạng `https://<ten-app>.streamlit.app/`.
