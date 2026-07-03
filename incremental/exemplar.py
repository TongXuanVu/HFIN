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
        class_data = np.asarray(class_data)
        if self.herding_pool and len(class_data) > self.herding_pool:
            rng = np.random.default_rng(int(class_label))
            pool_idx = rng.choice(len(class_data), size=self.herding_pool, replace=False)
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
        S = np.zeros(features.shape[1], dtype=np.float32)
        mask = np.zeros(len(class_data), dtype=bool)
        exemplar = []
        for k in range(1, m + 1):
            target_vector = k * class_mean - S
            scores = features @ target_vector      # (N,) — 1 matvec, nhẹ RAM
            scores[mask] = -np.inf
            best_idx = int(np.argmax(scores))
            mask[best_idx] = True
            S += features[best_idx]
            exemplar.append(class_data[best_idx])

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
