# HFIN: Hướng dẫn Cài đặt & Sử dụng

Tài liệu này cung cấp hướng dẫn chi tiết cách thiết lập môi trường và vận hành hệ thống **HFIN (Hierarchical Federated Class-Incremental Learning)**.

---

## 1. Yêu cầu Hệ thống
- **Hệ điều hành**: Windows/Linux/macOS.
- **Python**: 3.8 trở lên.
- **GPU**: Khuyến nghị NVIDIA GPU (CUDA) để huấn luyện nhanh (đặc biệt khi dùng 60 clients).

---

## 2. Thiết lập Môi trường

```bash
# Tạo môi trường ảo
conda create -n hfin python=3.10 -y
conda activate hfin

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 3. Chuẩn bị Dữ liệu (iNF-ToN-IoT & iNF-UQ-NIDS)

Dự án sử dụng các bộ dữ liệu NetFlow biến thể iNF (lấy từ bài báo).

### Bước 1: Vị trí Dữ liệu
Đặt các file CSV thô vào thư mục `data/raw/`.
Cấu trúc ví dụ:
```text
IDPS/
├── data/
│   ├── raw/
│   │   └── NF-ToN-IoT-v2.csv
│   │   └── NF-UQ-NIDS-v2.csv
```

### Bước 2: Phân chia Non-IID (Dirichlet)
Mặc định hệ thống sẽ tự động phân chia dữ liệu cho **60 Clients** theo thuật toán Dirichlet ($\alpha=0.5$). Mỗi **Edge Server** sẽ quản lý **20 Clients**.

---

## 4. Hướng dẫn Chạy (Running)

### ⚡ Chạy mặc định hệ thống (iNF-ToN-IoT-v2)
Lệnh này sẽ chạy 80 rounds toàn cục, với 5 rounds đánh giá định kỳ:
```bash
python main.py --dataset nf_ton_iot --epochs_global 80 --eval_interval 5
```

### 🔧 Tham số quan trọng
- `--dataset`: `nf_ton_iot` (10 lớp) hoặc `nf_uq_nids` (21 lớp).
- `--num_clients`: Số lượng thiết bị IIoT (Mặc định: 60).
- `--num_edge_servers`: Số lượng Edge Server (Mặc định: 3).
- `--epochs_global`: Số vòng tổng hợp tại Cloud (Bài báo khuyến nghị 40 hoặc 80).
- `--eval_interval`: Chu kỳ đánh giá model (Mặc định: 5).
- `--task_size`: Số lượng lớp mới học thêm mỗi task (VD: 2 hoặc 5).
- `--memory_size`: Bộ nhớ mẫu tại **mỗi Edge Server** (Mặc định: 500 mẫu).

### 🐛 Chế độ Thử nghiệm (Debug Mode)
Dùng để kiểm tra logic code nhanh với tập dữ liệu nhỏ:
```bash
python main.py --debug
```

---

## 5. Kết quả & Đánh giá (Evaluation)

Toàn bộ kết quả (accuracy, F1-score, forgetting metric) được lưu tại thư mục `logs/` theo thời gian thực.

### Các Chỉ số Chính:
1.  **Macro-F1**: Đánh giá khả năng nhận diện đa lớp trên dữ liệu mất cân bằng.
2.  **Average Forgetting ($f$)**: Đo lường mức độ quên kiến thức cũ sau khi học lớp mới. Công thức:
    $$f_k = \frac{1}{k-1} \sum_{j=1}^{k-1} (A_{j,j} - A_{k,j})$$
    (Trong đó $A_{k,j}$ là độ chính xác của lớp $j$ sau khi học task $k$).

### Đồ thị:
- `accuracy_curve.png`: Biểu đồ độ chính xác toàn cầu qua từng rounds.
- `confusion_matrix.png`: Ma trận nhầm lẫn chi tiết của task hiện tại.

---

## 💡 Lưu ý quan trọng từ Bài báo
- **WTO Selection**: Đừng ngạc nhiên nếu không phải tất cả Client đều gửi dữ liệu mỗi vòng. Edge Server chỉ chọn lọc các Client có dữ liệu quan trọng để tối ưu băng thông.
- **LR Setup**: Tốc độ học tự động chuyển từ $10^{-2}$ (Base Task) sang $2 \times 10^{-2}$ (Incremental Task) theo đúng Sec VI.B.
- **Edge Training**: Local Training hiện tại diễn ra tại Edge Server, không phải tại Client.
