"""
Cloud Server - Tầng trung tâm trong kiến trúc phân cấp HFIN
Tổng hợp global model từ các edge servers
Monitor hiệu năng global model
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.network import HFINNetwork
from data.dataset import NetFlowDataset
from federated.fed_utils import FedAvg, model_to_device


class CloudServer:
    """
    Cloud Server trung tâm trong HFIN
    - Tổng hợp models từ edge servers
    - Duy trì proxy dataset (reconstructed data)
    - Monitor model performance
    - Quản lý best model cho distillation
    """

    def __init__(self, num_classes, feature_extractor, device,
                 learning_rate=0.01, encode_model=None):
        """
        Args:
            num_classes: Số lớp ban đầu
            feature_extractor: MLPFeatureExtractor
            device: str
            learning_rate: LR cho reconstruction
            encode_model: Model nhỏ cho gradient inversion
        """
        self.device = device
        self.learning_rate = learning_rate
        self.model = HFINNetwork(num_classes, feature_extractor)
        self.encode_model = encode_model
        self.num_classes = num_classes

        # Monitor dataset (reconstructed data từ prototype gradients)
        self.monitor_data = []
        self.monitor_labels = []
        self.monitor_loader = None

        # Best models cho distillation
        self.best_model_1 = None  # Model tốt nhất trước task
        self.best_model_2 = None  # Model tốt nhất hiện tại
        self.best_perf = 0

    def aggregate_from_edges(self, edge_weights_list, sample_counts=None):
        """
        Tổng hợp global model từ các edge servers (Eq. 14)
        
        Args:
            edge_weights_list: list of state_dict từ edge servers
            sample_counts: list of int - số mẫu của mỗi edge
        
        Returns:
            new_global_weights: state_dict
        """
        if len(edge_weights_list) == 0:
            return self.model.state_dict()

        if sample_counts is not None and len(sample_counts) == len(edge_weights_list):
            # Weighted FedAvg theo Eq. 14 (Sử dụng hàm đã viết sẵn trong fed_utils)
            from federated.fed_utils import FedWeightedAvg
            global_weights = FedWeightedAvg(edge_weights_list, sample_counts)
        else:
            # Fallback to simple FedAvg
            global_weights = FedAvg(edge_weights_list)
            
        self.model.load_state_dict(global_weights)
        return global_weights

    def update_monitor(self, pool_grad=None):
        """
        Cập nhật monitor dataset và theo dõi performance
        
        Args:
            pool_grad: list of prototype gradients từ clients
        """
        # Reconstruction from gradients (nếu có)
        if pool_grad and len(pool_grad) > 0 and self.encode_model is not None:
            self._reconstruct_from_gradients(pool_grad)

        # Monitor performance
        if self.monitor_loader is not None:
            cur_perf = self._monitor_accuracy()
            print(f'  [Cloud] Monitor accuracy: {cur_perf:.2f}%')

            if cur_perf >= self.best_perf:
                self.best_perf = cur_perf
                self.best_model_1 = self.best_model_2
                self.best_model_2 = copy.deepcopy(self.model)
        else:
            # Nếu chưa có monitor data, lưu model hiện tại
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        """Trả về cặp old models cho distillation"""
        return [self.best_model_1, self.best_model_2]

    def get_global_model(self):
        """Trả về global model hiện tại"""
        return copy.deepcopy(self.model)

    def _monitor_accuracy(self):
        """Đánh giá model trên monitor dataset"""
        self.model.eval()
        correct, total = 0, 0

        for features, labels in self.monitor_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(features)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == labels).sum().item()
            total += len(labels)

        self.model.train()
        return 100.0 * correct / total if total > 0 else 0.0

    def _reconstruct_from_gradients(self, pool_grad):
        """
        Khôi phục dữ liệu từ prototype gradients
        (Gradient inversion - tương tự ProxyServer trong GLFC)
        """
        if self.encode_model is None:
            return

        # Xác định label từ gradients
        pool_label = []
        for w_single in pool_grad:
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1)
            pool_label.append(pred.item())

        # Reconstruct data từ mỗi gradient
        new_data = []
        new_labels = []
        
        # Lấy num_features từ encode_model
        if hasattr(self.encode_model, 'body') and len(self.encode_model.body) > 0:
            num_features = self.encode_model.body[0].in_features
        else:
            num_features = 43

        for i, (grad, label) in enumerate(zip(pool_grad, pool_label)):
            recon_model = copy.deepcopy(self.encode_model)
            recon_model = model_to_device(recon_model, self.device)

            # Dummy data khởi tạo ngẫu nhiên
            dummy_data = torch.randn((1, num_features)).to(self.device).requires_grad_(True)
            label_tensor = torch.LongTensor([label]).to(self.device)

            optimizer = optim.LBFGS([dummy_data], lr=0.1)
            criterion = nn.CrossEntropyLoss()

            # Tối ưu dummy data để match gradient
            n_iters = 100
            for it in range(n_iters):
                def closure():
                    optimizer.zero_grad()
                    pred = recon_model(dummy_data)
                    loss = criterion(pred, label_tensor)
                    dummy_grad = torch.autograd.grad(loss, recon_model.parameters(), create_graph=True)
                    grad_diff = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_grad, grad))
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)

            # Lưu reconstructed data
            reconstructed = dummy_data.detach().cpu().numpy()
            new_data.append(reconstructed.flatten())
            new_labels.append(label)

        if new_data:
            self.monitor_data = np.array(new_data, dtype=np.float32)
            self.monitor_labels = np.array(new_labels, dtype=np.int64)

            # Tạo monitor dataloader
            monitor_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.monitor_data),
                torch.LongTensor(self.monitor_labels)
            )
            self.monitor_loader = DataLoader(
                monitor_dataset, shuffle=True, batch_size=64, drop_last=False
            )
    def get_weights(self):
        """Trả về state_dict của global model hiện tại"""
        return copy.deepcopy(self.model.state_dict())