# HFIN Dataset Pipeline Summary

Tài liệu này tổng hợp toàn bộ các công việc và thông số kỹ thuật đã triển khai cho hệ thống dữ liệu của HFIN, đảm bảo khớp với các mô tả trong bài báo gốc.

## 1. Dữ liệu đầu vào (Datasets)

Hệ thống hỗ trợ 2 bộ dữ liệu mạng chuẩn NetFlow v2:
- **iNF-ToN-IoT-v2**: Tổng cộng 10 lớp (1 lớp Benign + 9 loại tấn công).
- **iNF-UQ-NIDS-v2**: Tổng cộng 21 lớp (1 lớp Benign + 20 loại tấn công từ 4 nguồn khác nhau).

## 2. Tự động hóa thu thập (Auto-Downloading)
- **File**: `data/downloader.py`
- **Tính năng**: 
  - Tự động kiểm tra và tải dữ liệu từ nhiều nguồn (HuggingFace, UQ eSpace).
  - Hỗ trợ resume download và giải nén tự động.
  - Sửa lỗi chọn file: Đảm bảo chọn đúng file CSV tương ứng với bộ dữ liệu yêu cầu.

## 3. Tiền xử lý (Preprocessing)
- **File**: `data/preprocessing.py`
- **Thông số kỹ thuật (Mục VI.B)**:
  - **Phân chia Train/Test**: 60% huấn luyện / 40% kiểm thử (Sử dụng `stratify` để giữ nguyên phân phối lớp).
  - **Features**: Trích xuất đúng **41 đặc trưng NetFlow** (loại bỏ chính xác 4-5 cột metadata tùy bộ dữ liệu).
  - **Chuẩn hóa**: Sử dụng `MinMaxScaler` (0-1).
  - **Cân bằng lớp**: Áp dụng kỹ thuật Downsampling để tránh mất cân bằng nhãn quá lớn giữa Benign và Attack.
- **Tối ưu hóa**: Sử dụng tham số `nrows` trong `pd.read_csv` khi cần lấy mẫu (max_samples), giúp xử lý các file dung lượng lớn (13GB) trong vài giây mà không tốn quá nhiều RAM.

## 4. Phân chia dữ liệu (Partitioning)
- **File**: `data/partition.py`
- **Cấu hình mạng (Mục VI.B)**:
  - **Edge Servers ($n$)**: 3 Edge.
  - **Clients ($m$)**: 20 Clients mỗi Edge (Tổng cộng **60 Clients**).
  - **Sắp xếp**: Phân chia client vào Edge Server theo khối (Edge 0 giữ Client 0-19, v.v.).
- **Mô phỏng Non-IID (Mục VI.B)**:
  - Sử dụng phân phối Dirichlet đặc chế $Poly(\alpha)$.
  - **$\alpha$ cho Benign**: 0.3 (Dữ liệu bình thường tập trung cao độ tại một số client).
  - **$\alpha$ cho Attack**: 0.8 (Dữ liệu tấn công phân tán rộng hơn).

## 5. Lịch trình học tăng cường (Task Schedule)
- **Base Task (Task 1)**: Khởi đầu với đúng **2 lớp** để xây dựng ranh giới phân loại ban đầu chính xác:
  - **ToN-IoT**: Benign (0) và Scanning (1).
  - **UQ-NIDS**: Benign (0) và DDoS (1).
- **Phân phối Task tiếp theo**:
  - **ToN-IoT (5 tasks)**: 2 + 2*4 = 10 lớp.
  - **UQ-NIDS (10 tasks)**: 2 + 2*9 = 20 lớp (lớp 21 được gộp vào task cuối).

## 6. Trực quan hóa (Visualization)
- **File**: `data/visualize_distribution.py`
- **Kết quả**:
  - **Heatmap**: Thể hiện số lượng mẫu của từng lớp trên mỗi Client.
  - **Stacked Bar Chart**: Thể hiện tỷ lệ phần trăm (distribution %) của các lớp trên toàn bộ 60 Clients.
  - Kết quả lưu tại thư mục `./plots/` và được tổng hợp trong `walkthrough.md`.

---
**Trạng thái**: Hoàn thành 100% phần Dataset. Sẵn sàng cho giai đoạn huấn luyện Federated Learning.
