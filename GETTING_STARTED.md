# HFIN: Hướng dẫn cài đặt và sử dụng

Tài liệu này hướng dẫn chi tiết cách tải dataset, thiết lập môi trường và các lệnh để chạy dự án **HFIN (Hierarchical Federated Class-Incremental Learning)**.

---

## 1. Yêu cầu hệ thống (Prerequisites)
- **Hệ điều hành**: Windows/Linux/macOS
- **Python**: Phiên bản 3.8 trở lên (Khuyến nghị 3.9 hoặc 3.10)
- **GPU**: Khuyến nghị có NVIDIA GPU + CUDA để huấn luyện nhanh hơn (vẫn hỗ trợ chạy trên CPU).

---

## 2. Thiết lập môi trường (Setup Environment)

Khuyến nghị sử dụng `conda` hoặc `venv` để tạo môi trường ảo độc lập.

### Cách 1: Sử dụng `conda` (Khuyến nghị)
```bash
# Tạo môi trường ảo tên 'hfin' với Python 3.10
conda create -n hfin python=3.10 -y

# Kích hoạt môi trường
conda activate hfin

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Cách 2: Sử dụng `venv` (Mặc định của Python)
```bash
# Tạo môi trường ảo
python -m venv hfin_env

# Kích hoạt môi trường (Windows)
hfin_env\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 3. Tải và chuẩn bị Dataset

Dự án HFIN mặc định sử dụng bộ dữ liệu mạng (Network Intrusion Detection). Các bộ dữ liệu được hỗ trợ:
1. **NF-UQ-NIDS-V2** (Mặc định)

### Bước 1: Trích xuất/Tải Dataset
Bạn cần tạo một thư mục `data/raw/` bên trong thư mục dự án và đặt file dataset vào đó.

Hệ thống hỗ trợ đọc file dưới định dạng `.csv` hoặc `.parquet`. **(Khuyến nghị dùng định dạng `.parquet` để tiết kiệm bộ nhớ và tăng tốc độ đọc dữ liệu).**

Cấu trúc thư mục mong muốn sau khi tải:
```text
HFIN/
├── data/
│   ├── raw/
│   │   └── NF-UQ-NIDS-V2.parquet   <-- (Hoặc .csv)
│   └── ...
```

### Bước 2: Tiền xử lý dữ liệu
Mặc dù `main.py` sẽ tự động gọi hàm tiền xử lý nếu chưa có dữ liệu xử lý, bạn cũng có thể chạy độc lập file `preprocessing.py` để kiểm tra dữ liệu trước:

```bash
# Chạy script tiền xử lý trực tiếp
python data/preprocessing.py ./data/raw/
```
Output mong đợi: Script sẽ tự tìm file `.parquet` hoặc `.csv` trong thư mục `raw`, cân bằng các nhãn (downsampling), mã hóa nhãn, chuẩn hóa dữ liệu, chia Train/Test và lưu kết quả ra file `data/raw/nf_unsw_nb15_processed.pkl`.

---

## 4. Các lệnh để chạy (How to Run)

Để bắt đầu quy trình huấn luyện Federated Class-Incremental Learning, chạy file `main.py`.

### ⚡ Lệnh chạy mặc định
Lệnh này sẽ chạy với cấu hình mặc định (bộ dữ liệu `nf_unsw_nb15`, 10 clients, 3 edge servers):
```bash
python main.py
```

### 🔧 Chạy với tham số tùy chỉnh
Bạn có thể tùy chỉnh các tham số huấn luyện thông qua command line. Dưới đây là ví dụ thay đổi một số cấu hình quan trọng:

```bash
python main.py \
    --dataset nf_unsw_nb15 \
    --data_path ./data/raw/ \
    --num_clients 20 \
    --num_edge_servers 4 \
    --epochs_global 100 \
    --epochs_local 5 \
    --batch_size 128 \
    --task_size 2 \
    --memory_size 2000 \
    --device cuda
```

### 🐛 Chạy thử nghiệm (Debug Mode)
Nếu bạn chỉ muốn chạy thử để kiểm tra code có lỗi hay không (không cần độ chính xác cao), hãy thêm flag `--debug`. Hệ thống sẽ giới hạn chỉ dùng 50,000 mẫu dữ liệu:
```bash
python main.py --debug
```

### Danh sách một số tham số quan trọng:
- `--dataset`: Tên bộ dữ liệu (`nf_unsw_nb15` hoặc `nf_ton_iot`).
- `--data_path`: Đường dẫn tới thư mục chứa file dữ liệu mẫu (`./data/raw/`).
- `--num_clients`: Số lượng client tham gia Federated Learning (Mặc định: 10).
- `--num_edge_servers`: Số lượng Edge Server (Mặc định: 3).
- `--epochs_global`: Số vòng (rounds) tổng hợp trên Cloud (Mặc định: 100).
- `--epochs_local`: Số epoch huấn luyện tại mỗi Local Client (Mặc định: 5).
- `--task_size`: Số lớp (classes) mới xuất hiện trong mỗi task học liên tục (Mặc định: 2).
- `--num_base_classes`: Số lớp khởi tạo ở Task 0 (Mặc định: 2).
- `--device`: Thiết bị chạy (`cuda` hoặc `cpu`).

---

## 5. Kết quả & Logs

Trong quá trình huấn luyện, tất cả trạng thái và kết quả sẽ tự động được lưu vào thư mục `logs/`. Mỗi lần chạy sẽ tạo một thư mục con chứa ngày giờ cụ thể (VD: `logs/20-03-26_18-05/`).

Các file sinh ra trong thư mục logs bao gồm:
1. `training.log`: Toàn bộ nhật ký console, kết quả vòng lặp.
2. `accuracy_curve.png`: Biểu đồ biểu diễn độ chính xác toàn cầu theo thời gian.
3. `confusion_matrix.png`: Biểu đồ ma trận nhầm lẫn của Task hiện tại.

Sau khi huấn luyện xong, mô hình Global tốt nhất (model checkpoint) sẽ được lưu tại `checkpoints/hfin_final_model.pth`.
