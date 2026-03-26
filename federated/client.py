import torch
import numpy as np
from data.dataset import NetFlowDataset

class HFINClient:
    """
    Federated Client cho HFIN - Đóng vai trò Data Provider (IIoT Device).
    Tài nguyên hạn chế, chỉ thu thập và cung cấp dữ liệu cho Edge Server.
    """

    def __init__(self, client_id, train_data, train_labels, device='cpu'):
        """
        Args:
            client_id: ID của client
            train_data: np.ndarray (N, features) - dữ liệu NetFlow local
            train_labels: np.ndarray (N,) - nhãn tương ứng
            device: str
        """
        self.client_id = client_id
        self.device = device

        # Dữ liệu local
        self.train_data = train_data
        self.train_labels = train_labels
        self.dataset = NetFlowDataset(train_data, train_labels)

    def get_class_counts(self):
        """Trả về phân bố lớp dữ liệu hiện tại (phục vụ WTO tại Server)"""
        unique_labels, counts = np.unique(self.train_labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))

    def get_data_for_edge(self, task_classes=None):
        """
        Cung cấp dữ liệu thô (flows) cho Edge Server.
        Nếu task_classes được chỉ định, chỉ trả về dữ liệu của các lớp đó.
        """
        if task_classes is None:
            return torch.FloatTensor(self.train_data), torch.LongTensor(self.train_labels)
        
        # Lọc dữ liệu theo task
        mask = np.isin(self.train_labels, task_classes)
        filtered_data = self.train_data[mask]
        filtered_labels = self.train_labels[mask]
        
        return torch.FloatTensor(filtered_data), torch.LongTensor(filtered_labels)


    def __len__(self):
        return len(self.train_data)
