import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from federated.fed_utils import FedAvg, model_to_device
from incremental.wto import wto_select_clients_for_data
from incremental.exemplar import ExemplarManager
from incremental.distillation import distillation_loss
from data.dataset import NetFlowDataset
from models.network import HFINNetwork

logger = logging.getLogger(__name__)


class EdgeServer:
    """
    Edge Server trong kiến trúc phân cấp HFIN.
    Đóng vai trò là "Local Trainer": tập hợp dữ liệu từ Clients,
    quản lý Exemplar Memory và thực hiện huấn luyện Class-Incremental (FCIL).
    """

    def __init__(self, edge_id, num_classes, feature_extractor, 
                 device='cpu', memory_size=500, task_size=2):
        """
        Args:
            edge_id: ID của edge server
            num_classes: Số lớp ban đầu
            feature_extractor: Backbone CNN
            device: str
            memory_size: Tổng số mẫu exemplar lưu tại Edge
            task_size: Số lớp mới mỗi task
        """
        self.edge_id = edge_id
        self.device = device
        self.task_size = task_size
        self.client_ids = []
        
        # Model & Incremental Learning
        self.model = HFINNetwork(num_classes, feature_extractor)
        self.old_model = None
        self.learned_classes = []
        
        # Exemplar Memory
        self.exemplar_manager = ExemplarManager(memory_size, feature_extractor.fc.in_features if hasattr(feature_extractor, 'fc') else 64)
        
        # Training state
        self.task_id_old = -1

    def set_clients(self, client_ids):
        """Gán danh sách clients thuộc phạm vi quản lý của Edge này"""
        self.client_ids = client_ids

    def train_local(self, clients_dict, global_round, task_id, 
                    task_classes, current_f1_scores, 
                    epochs=5, lr=0.01, batch_size=32, is_last_round=False):
        """
        Thực hiện huấn luyện cục bộ tại Edge Server (Giai đoạn 4).
        
        Args:
            clients_dict: Dict chứa các đối tượng HFINClient
            global_round: Round hiện tại
            task_id: ID của task incremental
            task_classes: List các lớp thuộc task hiện tại
            current_f1_scores: Dict F1 scores từ round trước (cho WTO)
            is_last_round: True nếu đây là round cuối của task hiện tại
                           → chỉ khi này mới cập nhật Exemplar Memory
        """
        # 1. WTO: Chọn lọc clients truyền dữ liệu dựa trên Priority
        client_infos = []
        for cid in self.client_ids:
            client = clients_dict[cid]
            client_infos.append({
                'client_id': cid,
                'class_counts': client.get_class_counts(),
                'transmission_rate': 10.0 # Giả định (Mbps), thực tế có thể lấy từ system
            })
            
        selected_client_ids = wto_select_clients_for_data(
            client_infos, current_f1_scores, beta=0.5
        )
        
        if not selected_client_ids:
            return self.get_weights()

        # 2. Thu thập và gộp dữ liệu từ các selected clients
        all_data = []
        all_labels = []
        for cid in selected_client_ids:
            data, labels = clients_dict[cid].get_data_for_edge(task_classes)
            all_data.append(data)
            all_labels.append(labels)
            
        X_train = torch.cat(all_data)
        y_train = torch.cat(all_labels)

        # 3. Phát hiện task mới — chỉ chạy 1 lần đầu task
        if task_id > self.task_id_old:
            self.task_id_old = task_id
            # Lưu model cũ để KD distillation
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            # Cập nhật learned_classes CHỈ 1 LẦN khi chuyển task mới
            # (không đặt ngoài block này vì sẽ bị gọi lặp mỗi round)
            self.learned_classes = list(range(
                len(task_classes) if task_id == 1
                else self.model.fc.out_features - len(task_classes)
            ))

        if self.old_model is not None:
            self.old_model.eval()

        # Kiểm tra nhãn dữ liệu có khớp với model không
        if len(y_train) > 0:
            max_label = int(y_train.max().item())
            if max_label >= self.model.fc.out_features:
                logger.warning(
                    f"Edge {self.edge_id}: Label {max_label} >= "
                    f"model out_features {self.model.fc.out_features}. "
                    f"Model expansion may be missing."
                )

        # 4. Mix với Exemplar Data (Replay) + Balanced Sampling
        exemplar_data, exemplar_labels = self.exemplar_manager.get_exemplar_data()
        is_old = [] # Track which samples are old for Selective KD
        
        # Nhãn mới từ client — lưu số lượng TRƯỚC khi concat
        num_new_samples = len(X_train)
        is_old.extend([False] * num_new_samples)
        
        if exemplar_data:
            all_x_exemplar = []
            all_y_exemplar = []
            for i, cls_samples in enumerate(exemplar_data):
                cls_x = np.array(cls_samples)
                all_x_exemplar.append(cls_x)
                all_y_exemplar.append(np.full(len(cls_samples), exemplar_labels[i]))
            
            flat_exemplar_data = torch.FloatTensor(np.concatenate(all_x_exemplar))
            flat_exemplar_labels = torch.LongTensor(np.concatenate(all_y_exemplar))
            
            # --- BALANCED REPLAY OPTIMIZATION ---
            # Nếu dữ liệu mới quá nhiều, oversample dữ liệu cũ để cân bằng Batch
            num_new = len(X_train)
            num_old = len(flat_exemplar_data)
            
            if num_old > 0 and num_new > num_old:
                repeat_factor = num_new // num_old
                if repeat_factor > 1:
                    logger.info(f"Edge {self.edge_id}: Oversampling memory by {repeat_factor}x to balance {num_new} new samples.")
                    flat_exemplar_data = flat_exemplar_data.repeat(repeat_factor, 1)
                    flat_exemplar_labels = flat_exemplar_labels.repeat(repeat_factor)
            
            is_old.extend([True] * len(flat_exemplar_data))
            X_train = torch.cat([X_train, flat_exemplar_data])
            y_train = torch.cat([y_train, flat_exemplar_labels])

        is_old = torch.BoolTensor(is_old)

        # 5. Training Loop
        # Chỉnh sửa NetFlowDataset để trả về cả is_old flag
        class BalancedDataset(torch.utils.data.Dataset):
            def __init__(self, x, y, is_old_mask):
                self.x = x
                self.y = y
                self.is_old = is_old_mask
            def __len__(self): return len(self.y)
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx], self.is_old[idx]

        dataset = BalancedDataset(X_train, y_train, is_old)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = model_to_device(self.model, self.device)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

        for epoch in range(epochs):
            for features, targets, mask in loader:
                features, targets, mask = features.to(self.device), targets.to(self.device), mask.to(self.device)

                outputs = self.model(features)

                # KD Loss (Selective: chỉ áp dụng cho mask=True)
                old_outputs = self.old_model(features) if self.old_model else None
                loss = distillation_loss(
                    outputs, old_outputs, targets,
                    self.model.fc.out_features,
                    self.old_model.fc.out_features if self.old_model else 0,
                    self.device,
                    is_old_mask=mask
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 6. Cập nhật Exemplar Memory sau khi huấn luyện xong task
        # *** BUG FIX: Chỉ chạy 1 lần vào round CUỐI của mỗi task ***
        # Nếu chạy mỗi round → memory bị reduce 80 lần/task → lớp cũ bị xóa sạch
        if is_last_round:
            # Trong HFIN/iCaRL, bộ nhớ mẫu được chia đều cho tất cả các lớp đã học
            # Tính toán số lượng mẫu cho mỗi lớp (m)
            total_learned_classes = len(self.exemplar_manager.exemplar_set) + len(task_classes)
            m = self.exemplar_manager.memory_size // total_learned_classes
            
            # Giảm số lượng mẫu của các lớp cũ
            self.exemplar_manager.reduce_exemplar_sets(m)
            
            # Thêm mẫu cho các lớp mới từ dữ liệu huấn luyện
            # Dùng num_new_samples (đã lưu trước concat) để tránh lấy exemplar cũ
            new_class_data_x = X_train[:num_new_samples]
            new_class_data_y = y_train[:num_new_samples]
            
            for cls in np.unique(new_class_data_y.cpu().numpy()):
                if cls in task_classes:
                    class_mask = (new_class_data_y == cls)
                    cls_data = new_class_data_x[class_mask].cpu().numpy()
                    self.exemplar_manager.construct_exemplar_set(
                        cls_data, int(cls), self.model, self.device, m=m
                    )
            logger.info(f"Edge {self.edge_id}: Memory updated at end of task. "
                        f"Total stored: {self.exemplar_manager.total_stored_samples} samples "
                        f"across {self.exemplar_manager.num_stored_classes} classes.")

        # Trả về đồng thời trọng số và số mẫu huấn luyện (cho FedWeightedAvg)
        return self.get_weights(), len(X_train)

    def get_weights(self):
        """Trả về state_dict của model"""
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights):
        """Cập nhật trọng số từ Cloud Server"""
        self.model.load_state_dict(weights)
