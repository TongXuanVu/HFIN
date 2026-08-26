"""
Quản lý Exemplar Memory cho Class-Incremental Learning trên dữ liệu tabular
Lưu giữ mẫu đại diện của lớp cũ để chống catastrophic forgetting
"""
import numpy as np
import torch
from torch.nn import functional as F


class ExemplarManager:
    """
    Quản lý bộ nhớ exemplar cho dữ liệu NetFlow tabular
    Tương tự exemplar management trong GLFC nhưng cho dữ liệu dạng bảng
    """

    def __init__(self, memory_size, feature_dim, herding_pool=200000):
        """
        Args:
            memory_size: Tổng số mẫu tối đa lưu trong bộ nhớ
            feature_dim: Chiều đặc trưng từ feature extractor
            herding_pool: Nếu 1 lớp có > herding_pool mẫu thì random lấy
                          herding_pool mẫu làm "pool" rồi mới herding trên pool
                          (tăng tốc ~N/pool lần). 0 = tắt (herding trên toàn bộ).
        """
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.herding_pool = herding_pool

        # exemplar_set[i] = list of samples (np.ndarray) cho lớp thứ i
        self.exemplar_set = []
        self.exemplar_labels = []  # nhãn tương ứng
        self.class_mean_set = []

    def pool_indices(self, n, class_label):
        """Chi so cua candidate pool cho mot lop co `n` mau, hoac None neu khong
        can cat (n <= herding_pool).

        Tach rieng ra de nguoi goi cat TREN CHI SO roi moi materialize mang —
        thay vi copy ca lop ra numpy roi mới cắt. Cung seed, cung `n`, nen
        pool_idx sinh ra y het cach cu: ket qua khong doi.
        """
        if not self.herding_pool or n <= self.herding_pool:
            return None
        rng = np.random.default_rng(int(class_label))
        return rng.choice(n, size=self.herding_pool, replace=False)

    def construct_exemplar_set(self, class_data, class_label, model, device, m=None):
        """
        Xây dựng exemplar set cho một lớp bằng herding selection
        Chọn m mẫu gần class mean nhất
        
        Args:
            class_data: np.ndarray (N, num_features) - dữ liệu của lớp
            class_label: int - nhãn lớp
            model: HFINNetwork - model hiện tại
            device: str
            m: int - số mẫu cần chọn (None = tự tính)
        """
        if m is None:
            total_classes = len(self.exemplar_set) + 1
            m = self.memory_size // total_classes

        # ── Candidate pooling: nếu lớp quá lớn, random lấy pool trước khi herding ──
        # Giảm chi phí herding (O(m×N)) và cả trích xuất feature. Seed theo class_label
        # để tái lập được. Vẫn giữ nguyên số exemplar m (buffer 1% không đổi) vì pool >= m.
        #
        # LUU Y: nguoi goi NEN dung pool_indices() de rut gon TRUOC khi
        # materialize mang numpy (xem edge_server.py). Nhanh duoi day chi con
        # la duong lui cho cac loi goi cu — no bat buoc phai giu ca lop trong
        # RAM truoc khi cat, voi client Benign 7,9 trieu mau la 976 MB.
        class_data = np.asarray(class_data)
        if self.herding_pool and len(class_data) > self.herding_pool:
            pool_idx = self.pool_indices(len(class_data), class_label)
            class_data = class_data[pool_idx]

        # Tính feature representations
        model.eval()
        features_list = []
        batch_size = 8192  # Cập nhật batch_size lớn hơn để trích xuất nhanh hơn
        with torch.no_grad():
            for i in range(0, len(class_data), batch_size):
                batch_data = class_data[i:i+batch_size]
                batch_x = torch.FloatTensor(batch_data).to(device)
                batch_features = model.feature_extractor(batch_x).cpu().numpy()
                features_list.append(batch_features)
            
            features = np.vstack(features_list)
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-8)

        # ── Herding selection kiểu iCaRL chuẩn (port từ AFSIC-IDS/utils/memory.py) ──
        # Dùng TÍCH VÔ HƯỚNG thay cho khoảng cách Euclid trên mảng (N,d):
        #   target = k*class_mean - S ; i = argmax( features · target )
        # Mỗi vòng chỉ cấp phát mảng `scores` dài N (không phải mảng N×d), nên
        # KHÔNG bị thrash bộ nhớ -> chạy xong được kể cả khi buffer lớn
        # (bản cũ của HFIN cấp phát N×d mỗi vòng -> treo khi m,N lớn).
        m = int(min(m, len(class_data)))

        if m >= len(class_data):
            # Giu TAT CA ung vien -> herding khong loc bo gi, chi doi thu tu, ma
            # thu tu khong duoc dung o dau (dataset.py gop bang np.concatenate).
            # Bo qua vong lap giup tiet kiem ~12 phut moi edge lon: voi
            # m = len = 200.000 thi vong lap cu ton 200.000 x 200.000 phep tinh
            # chi de sap xep lai mot tap ma cuoi cung van lay tron ven.
            exemplar = np.ascontiguousarray(class_data, dtype=np.float32)
        else:
            S = np.zeros(features.shape[1], dtype=np.float32)
            mask = np.zeros(len(class_data), dtype=bool)
            chon = np.empty(m, dtype=np.int64)
            for k in range(1, m + 1):
                target_vector = k * class_mean - S
                scores = features @ target_vector      # (N,) — 1 matvec, nhẹ RAM
                scores[mask] = -np.inf
                best_idx = int(np.argmax(scores))
                mask[best_idx] = True
                S += features[best_idx]
                chon[k - 1] = best_idx
            # Gom mot lan thay vi append tung hang: mot mang lien tuc (m, d) ton
            # ~124 B/mau, con list cac mang (d,) roi ton ~244 B/mau va tao ra m
            # object rieng le lam torch.save vua cham vua ngon gap doi RAM.
            exemplar = np.ascontiguousarray(class_data[chon], dtype=np.float32)

        self.exemplar_set.append(exemplar)
        self.exemplar_labels.append(class_label)

    def reduce_exemplar_sets(self, m):
        """Giảm kích thước mỗi exemplar set xuống m mẫu"""
        for i in range(len(self.exemplar_set)):
            self.exemplar_set[i] = self.exemplar_set[i][:m]

    def get_exemplar_data(self):
        """Trả về tất cả exemplar dưới dạng arrays"""
        if len(self.exemplar_set) == 0:
            return None, None
        return self.exemplar_set, self.exemplar_labels

    @property
    def num_stored_classes(self):
        return len(self.exemplar_set)

    @property 
    def total_stored_samples(self):
        return sum(len(e) for e in self.exemplar_set)
