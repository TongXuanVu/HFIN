"""
HFIN Client - Huấn luyện local trên thiết bị IIoT
Chuyển đổi từ GLFC_model sang dữ liệu tabular NetFlow
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.network import HFINNetwork
from data.dataset import NetFlowDataset
from incremental.exemplar import ExemplarManager
from incremental.distillation import (
    distillation_loss, efficient_old_class_weight, get_one_hot
)
from federated.fed_utils import model_to_device


class HFINClient:
    """
    Federated Client cho HFIN
    Mỗi client đại diện cho một thiết bị IIoT với dữ liệu local
    """

    def __init__(self, client_id, num_classes, feature_extractor,
                 batch_size, task_size, memory_size, epochs,
                 learning_rate, train_data, train_labels, device,
                 num_base_classes=4, encode_model=None):
        """
        Args:
            client_id: ID của client
            num_classes: Số lớp ban đầu
            feature_extractor: MLPFeatureExtractor
            batch_size: Batch size
            task_size: Số lớp mới mỗi task
            memory_size: Kích thước exemplar memory
            epochs: Số epochs local
            learning_rate: Learning rate
            train_data: np.ndarray (N, features) - dữ liệu training local
            train_labels: np.ndarray (N,) - nhãn
            device: str
            num_base_classes: Số lớp base task
            encode_model: Model nhỏ cho prototype gradient
        """
        self.client_id = client_id
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.task_size = task_size
        self.num_base_classes = num_base_classes
        self.device = device

        # Model
        self.model = HFINNetwork(num_classes, feature_extractor)
        self.encode_model = encode_model
        self.old_model = None

        # Dữ liệu
        self.train_dataset = NetFlowDataset(train_data, train_labels)
        self.train_loader = None

        # Exemplar
        self.exemplar_manager = ExemplarManager(memory_size, feature_dim=64)

        # State
        self.num_classes = 0
        self.learned_numclass = 0
        self.learned_classes = []
        self.current_classes = None
        self.last_classes = None
        self.task_id_old = -1
        self.signal = False  # Tín hiệu entropy (phát hiện task mới)
        self.last_entropy = 0

    def beforeTrain(self, task_id, group):
        """
        Chuẩn bị dữ liệu cho task mới
        
        Args:
            task_id: ID task hiện tại
            group: 0 = old client (không nhận lớp mới), 1 = client nhận lớp mới
        """
        if task_id != self.task_id_old:
            self.task_id_old = task_id
            self.signal = True
            # Lấy số lớp trực tiếp từ đầu ra của mô hình (đã được cap bởi total_classes ở Server)
            self.num_classes = self.model.fc.out_features

            if group != 0:
                self.last_classes = self.current_classes
                if task_id == 0:
                    self.current_classes = list(range(self.num_base_classes))
                else:
                    start = self.num_base_classes + (task_id - 1) * self.task_size
                    end = start + self.task_size
                    self.current_classes = list(range(start, end))
            else:
                self.last_classes = None
        else:
            self.signal = False

        # Load dữ liệu cho lớp hiện tại (không có exemplar)
        self.train_dataset.getTrainData(self.current_classes)
        if len(self.train_dataset) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_dataset, shuffle=True,
                batch_size=self.batch_size, num_workers=0, pin_memory=False
            )
        else:
            self.train_loader = None

    def update_exemplar(self):
        """
        Cập nhật exemplar set:
        1. Nhận tín hiệu task mới từ Server thông qua `task_id` (chính xác 100% theo bài báo)
        2. Nếu có task mới: lưu exemplar cho lớp cũ
        3. Refresh training data với exemplar
        """
        self.model = model_to_device(self.model, self.device)
        self.model.eval()

        if self.signal and self.last_classes is not None:
            self.learned_numclass += len(self.last_classes)
            self.learned_classes.extend(self.last_classes)

            # Giảm exemplar set cũ
            m = self.exemplar_manager.memory_size // max(1, self.learned_numclass)
            self.exemplar_manager.reduce_exemplar_sets(m)

            # Thêm exemplar cho lớp vừa học xong
            for cls in self.last_classes:
                class_data = self.train_dataset.get_class_data(cls)
                if len(class_data) > 0:
                    self.exemplar_manager.construct_exemplar_set(
                        class_data, cls, self.model, self.device, m
                    )

        self.model.train()

        # Refresh training data với exemplar
        exemplar_data, exemplar_labels = self.exemplar_manager.get_exemplar_data()
        self.train_dataset.getTrainData(
            self.current_classes, exemplar_data, exemplar_labels
        )
        if len(self.train_dataset) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_dataset, shuffle=True,
                batch_size=self.batch_size, num_workers=0, pin_memory=False
            )
        else:
            self.train_loader = None

    def train(self, ep_g, model_old_pair):
        """
        Huấn luyện local
        
        Args:
            ep_g: Global round hiện tại
            model_old_pair: [old_model_1, old_model_2] từ proxy/cloud server
        """
        if self.train_loader is None or len(self.train_dataset) == 0:
            return

        self.model = model_to_device(self.model, self.device)
        opt = optim.SGD(self.model.parameters(),
                        lr=self.learning_rate, weight_decay=1e-5)

        # Chọn old model để distillation
        if model_old_pair[1] is not None:
            self.old_model = model_old_pair[1] if self.signal else model_old_pair[0]
        elif self.signal:
            self.old_model = model_old_pair[0]

        if self.old_model is not None:
            self.old_model = model_to_device(self.old_model, self.device)
            self.old_model.eval()

        # Training loop
        for epoch in range(self.epochs):
            # Learning rate schedule
            self._adjust_lr(opt, epoch, ep_g)

            for step, (_, features, target) in enumerate(self.train_loader):
                features = features.to(self.device)
                target = target.to(self.device)

                loss = self._compute_loss(features, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _compute_loss(self, features, label):
        """Tính loss = weighted CE + knowledge distillation"""
        output = self.model(features)
        target = get_one_hot(label, self.num_classes, self.device)

        if self.old_model is None:
            # Chỉ CE loss với class weight
            w = efficient_old_class_weight(
                output, label, self.num_classes, self.learned_classes, self.device
            )
            loss = torch.mean(w * F.binary_cross_entropy_with_logits(
                output, target, reduction='none'
            ))
            return loss
        else:
            # CE + distillation
            w = efficient_old_class_weight(
                output, label, self.num_classes, self.learned_classes, self.device
            )
            loss_ce = torch.mean(w * F.binary_cross_entropy_with_logits(
                output, target, reduction='none'
            ))

            # Distillation target
            distill_target = target.clone()
            old_output = torch.sigmoid(self.old_model(features))
            old_size = old_output.shape[1]
            distill_target[..., :old_size] = old_output
            loss_kd = F.binary_cross_entropy_with_logits(output, distill_target)

            return 0.5 * loss_ce + 0.5 * loss_kd



    def _adjust_lr(self, optimizer, epoch, ep_g):
        """Điều chỉnh learning rate"""
        combined_epoch = epoch + ep_g * self.epochs
        if combined_epoch % 200 == 100:
            for p in optimizer.param_groups:
                p['lr'] = self.learning_rate / 5
        elif combined_epoch % 200 == 150:
            for p in optimizer.param_groups:
                p['lr'] = self.learning_rate / 25
        elif combined_epoch % 200 == 180:
            for p in optimizer.param_groups:
                p['lr'] = self.learning_rate / 125

    def get_model_weights(self):
        """Trả về state_dict của model"""
        return copy.deepcopy(self.model.state_dict())

    def get_class_counts(self):
        """Trả về phân bố lớp dữ liệu hiện tại (cho WTO)"""
        return self.train_dataset.get_class_count()

    def proto_grad_sharing(self):
        """Chia sẻ prototype gradient (cho reconstruction trên server)"""
        if not self.signal or self.current_classes is None:
            return None
        return self._prototype_mask()

    def _prototype_mask(self):
        """Tạo prototype gradient cho các lớp hiện tại"""
        proto_grad = []
        self.model.eval()

        for cls in self.current_classes:
            class_data = self.train_dataset.get_class_data(cls)
            if len(class_data) == 0:
                continue

            # Tìm mẫu gần class mean nhất (prototype)
            with torch.no_grad():
                x = torch.FloatTensor(class_data).to(self.device)
                feat = self.model.feature_extractor(x).cpu().numpy()
                feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-10)

            class_mean = np.mean(feat, axis=0)
            dist = np.linalg.norm(class_mean - feat, axis=1)
            proto_idx = np.argmin(dist)

            # Tính gradient qua encode model
            if self.encode_model is not None:
                self.encode_model = model_to_device(self.encode_model, self.device)
                data = torch.FloatTensor(class_data[proto_idx:proto_idx+1]).to(self.device)
                label = torch.LongTensor([cls]).to(self.device)

                criterion = nn.CrossEntropyLoss()
                outputs = self.encode_model(data)
                loss = criterion(outputs, label)
                dy_dx = torch.autograd.grad(loss, self.encode_model.parameters())
                proto_grad.append(list(_.detach().clone() for _ in dy_dx))

        return proto_grad if proto_grad else None
