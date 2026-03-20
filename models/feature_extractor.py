"""
Feature Extractor MLP cho dữ liệu NetFlow tabular
Thay thế ResNet18-CBAM (dùng cho ảnh) trong GLFC gốc
"""
import torch
import torch.nn as nn


class MLPFeatureExtractor(nn.Module):
    """
    Multi-Layer Perceptron cho trích xuất đặc trưng từ dữ liệu NetFlow.
    Input: (batch, num_features) → Output: (batch, feature_dim)
    """

    def __init__(self, input_dim=43, hidden_dims=None, output_dim=64, dropout=0.3):
        """
        Args:
            input_dim: Số features đầu vào (43 cho NetFlow v2)
            hidden_dims: List kích thước các lớp ẩn (mặc định [128, 256, 128])
            output_dim: Chiều đặc trưng đầu ra
            dropout: Tỷ lệ dropout
        """
        super(MLPFeatureExtractor, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        # Lớp output
        layers.append(nn.Linear(prev_dim, output_dim))

        self.body = nn.Sequential(*layers)

        # Thuộc tính fc để tương thích với network.py (lấy in_features)
        self.fc = nn.Linear(output_dim, output_dim)  # placeholder

    def forward(self, x):
        return self.body(x)


class LeNetTabular(nn.Module):
    """
    Mạng nhỏ dùng cho encode_model (tương tự LeNet trong GLFC,
    nhưng cho dữ liệu tabular thay vì ảnh)
    """

    def __init__(self, input_dim=43, hidden_dim=128, num_classes=10):
        super(LeNetTabular, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = self.fc(out)
        return out


def weights_init(m):
    """Khởi tạo trọng số ngẫu nhiên (tương tự GLFC)"""
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
    try:
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
