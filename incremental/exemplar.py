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

    def __init__(self, memory_size, feature_dim):
        """
        Args:
            memory_size: Tổng số mẫu tối đa lưu trong bộ nhớ
            feature_dim: Chiều đặc trưng từ feature extractor
        """
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
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

        # Tính feature representations
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(class_data).to(device)
            features = model.feature_extractor(x).cpu().numpy()
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

        class_mean = np.mean(features, axis=0)

        # Herding selection: chọn m mẫu sao cho mean gần class mean nhất
        exemplar = []
        exemplar_features = np.zeros((1, features.shape[1]))
        
        selected_indices = set()
        for i in range(min(m, len(class_data))):
            # Tìm mẫu minimizes khoảng cách giữa running mean và class mean
            running_mean = exemplar_features / (i + 1)
            candidate_means = class_mean - (running_mean + features) / (i + 1)
            distances = np.linalg.norm(candidate_means, axis=1)
            
            # Loại bỏ indices đã chọn
            for idx in selected_indices:
                distances[idx] = float('inf')
            
            best_idx = np.argmin(distances)
            selected_indices.add(best_idx)
            exemplar_features += features[best_idx:best_idx+1]
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
