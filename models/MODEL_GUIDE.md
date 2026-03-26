# HFIN Model Architecture Summary (MODEL_GUIDE.md)

Tài liệu này tổng hợp các chi tiết kỹ thuật về kiến trúc mô hình đã được xây dựng trong Giai đoạn 2, đảm bảo khớp hoàn toàn với đặc tả của bài báo HFIN.

## 1. Feature Extractor (Bộ trích xuất đặc trưng)
- **File**: `models/feature_extractor.py`
- **Kiến trúc**: 1-D CNN (phù hợp cho dữ liệu mạng dạng tabular).
- **Cấu hình chi tiết (Khớp Fig 4)**:
  - Gồm đúng **4 lớp Conv1d** nối tiếp nhau.
  - Thứ tự các lớp: **Conv1d -> ReLU -> BatchNorm1d**.
  - Các tham số: `kernel_size=3`, `stride=1`, `padding=0`.
  - Downsampling: Sử dụng `MaxPool1d(2, 2)` sau lớp thứ 2.
  - Global Pooling: `AdaptiveMaxPool1d(1)` ở cuối cùng.
- **Dữ liệu đầu ra**: Vector 64 chiều từ lớp BatchNorm cuối cùng, đi trực tiếp vào Adaptive Classification Head (Loại bỏ lớp Linear trung gian).

## 2. Adaptive Classifier Head (Lớp phân loại thích ứng)
- **File**: `models/network.py`
- **Mục tiêu**: Cho phép học tăng cường (Incremental Learning) mà không mất tri thức cũ.
- **Cơ chế**:
  - Khi có lớp tấn công mới (new task), mô hình sẽ tự động mở rộng (concatenate) các node mới vào lớp Linear cuối cùng.
  - **Bảo tồn trọng số**: Copy toàn bộ `weight` và `bias` từ task cũ sang task mới, đảm bảo tính kế thừa kiến thức.

## 3. Weight Aligning (WA - Hiệu chỉnh trọng số)
- **File**: `models/network.py` (Hàm `weight_align`)
- **Vấn đề giải quyết**: Chống lại sự "quên" (Forgetting) và thiên kiến (Bias) về phía các lớp mới.
- **Logic**:
  - Tính toán độ dài (Norm) trung bình của các vector trọng số lớp cũ so với lớp mới.
  - Hiệu chỉnh trọng số của lớp mới bằng hệ số `gamma` ($\gamma = \text{MeanNorm}_{\text{old}} / \text{MeanNorm}_{\text{new}}$).
  - Kết quả: Mô hình duy trì khả năng nhận diện cân bằng giữa tấn công cũ và mới.

---
**Trạng thái**: Hoàn thành Giai đoạn 2. Mô hình đã sẵn sàng cho logic huấn luyện thực tế.
