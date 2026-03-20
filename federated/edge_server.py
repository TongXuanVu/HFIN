"""
Edge Server - Tầng trung gian trong kiến trúc phân cấp HFIN
Tổng hợp model updates từ các clients trong nhóm + áp dụng WTO
"""
import copy
import torch
from federated.fed_utils import FedAvg, FedWeightedAvg
from incremental.wto import compute_transmission_weights


class EdgeServer:
    """
    Edge Server trong kiến trúc phân cấp HFIN
    Mỗi edge server quản lý một nhóm clients,
    tổng hợp updates cục bộ rồi gửi lên cloud server
    """

    def __init__(self, edge_id, device='cpu'):
        """
        Args:
            edge_id: ID của edge server
            device: str
        """
        self.edge_id = edge_id
        self.device = device
        self.client_ids = []  # Danh sách client thuộc edge này
        self.aggregated_weights = None

    def set_clients(self, client_ids):
        """Gán danh sách clients cho edge server"""
        self.client_ids = client_ids

    def aggregate(self, client_models_info, use_wto=True, wto_alpha=0.5, max_time=2.0):
        """
        Tổng hợp model updates từ các clients
        
        Args:
            client_models_info: list of dict với keys:
                - 'client_id': int
                - 'model_weights': state_dict
                - 'class_counts': dict {class_id: count}
            use_wto: bool - có áp dụng WTO không
            wto_alpha: float - tham số WTO
            max_time: float - giới hạn thời gian (áp dụng Shannon-Hartley)
        
        Returns:
            aggregated_weights: state_dict
        """
        if len(client_models_info) == 0:
            return None

        if use_wto and len(client_models_info) > 1:
            # Áp dụng WTO: chọn clients quan trọng + weighted avg
            selected_weights, agg_weights = compute_transmission_weights(
                client_models_info, alpha=wto_alpha, max_time=max_time
            )

            if len(selected_weights) > 0:
                self.aggregated_weights = FedWeightedAvg(selected_weights, agg_weights)
            else:
                # Fallback: FedAvg bình thường
                all_weights = [info['model_weights'] for info in client_models_info]
                self.aggregated_weights = FedAvg(all_weights)
        else:
            # FedAvg bình thường
            all_weights = [info['model_weights'] for info in client_models_info]
            self.aggregated_weights = FedAvg(all_weights)

        return self.aggregated_weights

    def get_aggregated_weights(self):
        """Trả về model đã tổng hợp để gửi lên cloud"""
        return self.aggregated_weights
