"""
PyTorch Dataset cho dữ liệu NetFlow - hỗ trợ Class-Incremental Learning
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class NetFlowDataset(Dataset):
    """
    Dataset cho dữ liệu NetFlow tabular.
    Hỗ trợ:
    - Load toàn bộ hoặc theo danh sách lớp
    - Thêm exemplar set (dữ liệu cũ giữ lại)
    - Lấy dữ liệu theo lớp cụ thể
    """

    def __init__(self, X, y):
        """
        Args:
            X: np.ndarray hoặc torch.Tensor (N, num_features)
            y: np.ndarray hoặc torch.Tensor (N,)
        """
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
            
        self.X_all = X.astype(np.float32)
        self.y_all = y.astype(np.int64)

        # Dữ liệu hiện tại (sẽ được filter theo task)
        self.X = self.X_all.copy()
        self.y = self.y_all.copy()
        self._sync_tensors()

    def _sync_tensors(self):
        """
        Tao san tensor tu self.X / self.y (goi moi khi self.X/self.y thay doi).
        Giup __getitem__ chi can index thay vi tao FloatTensor moi moi mau
        -> tang toc DataLoader ma KHONG doi gia tri (view chung buffer numpy).
        """
        self._X_t = torch.from_numpy(np.ascontiguousarray(self.X, dtype=np.float32))
        self._y_t = torch.from_numpy(np.ascontiguousarray(self.y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return idx, self._X_t[idx], self._y_t[idx]

    def getTrainData(self, classes, exemplar_set=None, exemplar_classes=None):
        """
        Lấy dữ liệu train cho các lớp được chỉ định + exemplar
        
        Args:
            classes: list[int] - danh sách lớp hiện tại cần train
            exemplar_set: list[np.ndarray] - exemplar data cho mỗi lớp cũ
            exemplar_classes: list[int] - nhãn lớp tương ứng với exemplar_set
        """
        if classes is None:
            self.X = self.X_all.copy()
            self.y = self.y_all.copy()
            self._sync_tensors()
            return

        # Lọc dữ liệu theo classes hiện tại
        mask = np.isin(self.y_all, classes)
        X_current = self.X_all[mask]
        y_current = self.y_all[mask]

        # Gộp với exemplar nếu có
        if exemplar_set and len(exemplar_set) > 0 and exemplar_classes and len(exemplar_classes) > 0:
            X_exemplar_list = []
            y_exemplar_list = []
            for exemplar_data, exemplar_label in zip(exemplar_set, exemplar_classes):
                if len(exemplar_data) > 0:
                    X_exemplar_list.append(np.array(exemplar_data, dtype=np.float32))
                    y_exemplar_list.append(np.full(len(exemplar_data), exemplar_label, dtype=np.int64))
            
            if X_exemplar_list:
                X_exemplar = np.concatenate(X_exemplar_list, axis=0)
                y_exemplar = np.concatenate(y_exemplar_list, axis=0)
                self.X = np.concatenate([X_current, X_exemplar], axis=0)
                self.y = np.concatenate([y_current, y_exemplar], axis=0)
            else:
                self.X = X_current
                self.y = y_current
        else:
            self.X = X_current
            self.y = y_current
        self._sync_tensors()

    def getTestData(self, class_range):
        """
        Lấy dữ liệu test cho một phạm vi lớp [start, end)
        
        Args:
            class_range: [start, end] - phạm vi lớp cần test
        """
        if isinstance(class_range, list) and len(class_range) == 2:
            classes = list(range(class_range[0], class_range[1]))
        else:
            classes = class_range
        
        mask = np.isin(self.y_all, classes)
        self.X = self.X_all[mask]
        self.y = self.y_all[mask]
        self._sync_tensors()

    def get_class_data(self, class_id):
        """Lấy tất cả dữ liệu của một lớp cụ thể"""
        mask = self.y_all == class_id
        return self.X_all[mask]

    def get_class_count(self):
        """Đếm số mẫu mỗi lớp"""
        unique, counts = np.unique(self.y_all, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @property
    def num_features(self):
        return self.X_all.shape[1]

    @property
    def num_classes(self):
        return len(np.unique(self.y_all))
