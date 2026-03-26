# WTO (Weighted Transmission Optimization) Guide

Tài liệu này giải thích cơ chế WTO giúp Edge Server chọn lọc DỮ LIỆU từ các Client (thiết bị IIoT) để tối ưu hóa băng thông trong hệ thống HFIN.

## 1. Vai trò của WTO trong IIoT
Vì các thiết bị IIoT (Client) có tài nguyên hạn chế, chúng **KHÔNG** tự huấn luyện mô hình.
- **Client**: Thu thập dữ liệu Flows và gửi yêu cầu truyền tải lên Edge Server.
- **Edge Server**: Sử dụng WTO để quyết định Client nào được phép truyền dữ liệu "quan trọng" nhất lên để huấn luyện, nhằm tránh nghẽn băng thông mạng IIoT.

## 2. Công thức cốt lõi

### Mức độ quan trọng của dữ liệu (Importance - Eq. 8)
$$w_k^i = \frac{F1_k^*}{F1_k^t \cdot (Num_k^i)^\beta}$$
Trong đó:
- $F1_k^*$: Ngưỡng F1 mục tiêu (mặc định 0.95).
- $F1_k^t$: Chỉ số F1 hiện tại của lớp $k$ (từ Edge/Cloud).
- $Num_k^i$: Số lượng mẫu của lớp $k$ tại Client $i$.
- $\beta$: Tham số điều tiết (mặc định 0.5).

### Độ ưu tiên truyền tải (Priority - Eq. 9)
$$P_i = \text{Importance}_i \times \text{Rate}_i$$
Trong đó:
- $\text{Rate}_i$: Tốc độ truyền tải Shannon-Hartley (Mbps), tính dựa trên SNR và khoảng cách.

## 3. Quy trình thực hiện
1. **Client** gửi metadata về số lượng mẫu các lớp (`class_counts`).
2. **Edge Server** tính toán **Importance** cho từng Client dựa trên trạng thái F1 hiện tại.
3. **Edge Server** ước tính **Transmission Rate** (Shannon-Hartley).
4. Sắp xếp danh sách Client bằng **Bubble Sort** theo $P_i$ giảm dần (yêu cầu đặc thù của bài báo).
5. Chọn ra các Client được phép gửi dữ liệu thô (Raw Data) lên Edge để bắt đầu huấn luyện local.

## 4. File liên quan
- [wto.py](file:///d:/IDPS/HFIN-core/IDPS/incremental/wto.py): Chứa toàn bộ logic tính toán và sắp xếp.
