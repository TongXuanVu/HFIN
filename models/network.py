"""
Mạng phân loại cho HFIN - Feature Extractor + Classification Head
Hỗ trợ mở rộng head khi có lớp tấn công mới (Class-Incremental)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class HFINNetwork(nn.Module):
    """
    Mạng kết hợp: Feature Extractor + Incremental Classifier
    Tương tự class `network` trong GLFC nhưng cho dữ liệu tabular
    """

    def __init__(self, num_classes, feature_extractor):
        """
        Args:
            num_classes: Số lớp ban đầu
            feature_extractor: Module trích xuất đặc trưng (MLPFeatureExtractor)
        """
        super(HFINNetwork, self).__init__()
        self.feature = feature_extractor

        # Lấy output dim từ feature extractor
        # Tìm lớp Linear cuối cùng trong body
        feature_dim = self._get_feature_dim()
        self.fc = nn.Linear(feature_dim, num_classes, bias=True)

    def _get_feature_dim(self):
        """Tìm output dimension của feature extractor"""
        for module in reversed(list(self.feature.body.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        # Fallback
        return self.feature.fc.in_features

    def forward(self, x):
        """Forward pass: features → logits"""
        features = self.feature(x)
        logits = self.fc(features)
        return logits

    def Incremental_learning(self, new_num_classes):
        """
        Mở rộng classification head cho lớp mới
        Giữ nguyên trọng số cũ, thêm neurons mới cho lớp mới
        """
        old_weight = self.fc.weight.data
        old_bias = self.fc.bias.data
        in_features = self.fc.in_features
        old_num_classes = self.fc.out_features

        # Tạo FC layer mới lớn hơn
        self.fc = nn.Linear(in_features, new_num_classes, bias=True)
        # Copy trọng số cũ
        self.fc.weight.data[:old_num_classes] = old_weight
        self.fc.bias.data[:old_num_classes] = old_bias

    def feature_extractor(self, x):
        """Trích xuất đặc trưng (không qua FC head)"""
        return self.feature(x)

    def predict(self, features):
        """Phân loại từ features"""
        return self.fc(features)
