"""
WTO - Weighted Transmission Optimization
Đóng góp chính của bài báo HFIN:
- Ưu tiên truyền model updates từ clients có dữ liệu tấn công hiếm/quan trọng
- Cân bằng bandwidth với khả năng phát hiện toàn diện
"""
import numpy as np
import torch
import random


def calculate_shannon_hartley_rate(B_mhz=20, P_dbm_min=15, P_dbm_max=23, N0_dbm=-100, path_loss=80):
    """
    Tính tốc độ truyền tải theo lý thuyết Shannon-Hartley (R = B * log2(1 + S/N))
    Mô phỏng mạng vô tuyến IIoT.
    """
    # Random công suất phát cho client để tạo sự đa dạng
    P_dbm = random.uniform(P_dbm_min, P_dbm_max)
    
    P_watt = 10**((P_dbm - 30)/10)
    N0_watt = 10**((N0_dbm - 30)/10)
    PL_linear = 10**(path_loss/10)
    
    signal_power = P_watt / PL_linear
    snr = signal_power / N0_watt
    
    B_hz = B_mhz * 1e6
    rate_bps = B_hz * np.log2(1 + snr)
    return rate_bps / 1e6  # Trả về Mbps


def compute_class_importance(client_class_counts, all_class_counts, alpha=0.5):
    """
    Tính trọng số quan trọng cho mỗi client dựa trên phân bố lớp
    
    Ý tưởng WTO: Clients có dữ liệu tấn công hiếm (ít xuất hiện trong toàn cục)
    nên được ưu tiên truyền model updates hơn.
    
    importance = alpha * (1 / frequency) + (1 - alpha) * diversity_score
    
    Args:
        client_class_counts: dict {class_id: count} - phân bố lớp của client
        all_class_counts: dict {class_id: count} - phân bố lớp toàn cục
        alpha: float - cân bằng giữa rarity vs diversity
    
    Returns:
        importance_score: float
    """
    if not client_class_counts:
        return 0.0
    
    total_global = sum(all_class_counts.values())
    
    # 1. Rarity score: lớp càng hiếm trong toàn cục → score càng cao
    rarity_score = 0.0
    for cls, count in client_class_counts.items():
        global_freq = all_class_counts.get(cls, 1) / total_global
        # Inverse frequency: lớp hiếm có trọng số cao hơn
        rarity_score += count * (1.0 / (global_freq + 1e-10))
    
    total_client = sum(client_class_counts.values())
    rarity_score = rarity_score / (total_client + 1e-10)
    
    # 2. Diversity score: client có nhiều lớp khác nhau → score cao hơn
    num_classes_total = len(all_class_counts)
    num_classes_client = len(client_class_counts)
    diversity_score = num_classes_client / (num_classes_total + 1e-10)
    
    # Kết hợp
    importance = alpha * rarity_score + (1 - alpha) * diversity_score
    
    return importance


def wto_select_clients(client_importance_scores, max_time_seconds=2.0, model_size_mb=5.0, min_clients=2):
    """
    Chọn clients để truyền model updates dựa trên WTO và Băng thông thực tế (Shannon-Hartley)
    Ưu tiên clients có importance score cao, nhưng phải thỏa mãn constraint về thời gian truyền.
    
    Args:
        client_importance_scores: dict {client_id: importance_score}
        max_time_seconds: float - giới hạn thời gian tổng (hoặc cho mỗi client tùy định nghĩa)
        model_size_mb: float - kích thước model (Megabytes)
        min_clients: int - số clients tối thiểu phải chọn
    
    Returns:
        selected_clients: list of client_ids
        client_weights: dict {client_id: weight} cho FedWeightedAvg
    """
    if len(client_importance_scores) == 0:
        return [], {}
    
    # Sắp xếp theo importance giảm dần
    sorted_clients = sorted(client_importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    selected_clients = []
    selected_scores = []
    
    # Mô phỏng chọn client dựa trên constraint Shannon-Hartley
    for client_id, imp in sorted_clients:
        # Tính R (Mbps)
        R_mbps = calculate_shannon_hartley_rate()
        
        # Tính thời gian truyền model (Model Size tính bằng Megabits = MB * 8)
        model_size_mbits = model_size_mb * 8
        time_required = model_size_mbits / (R_mbps + 1e-5)
        
        # Nếu thiết bị có thời gian truyền tốt (trong ngưỡng cho phép), chọn client này
        if time_required <= max_time_seconds or len(selected_clients) < min_clients:
            selected_clients.append(client_id)
            selected_scores.append(imp)
            
            # Giới hạn số lượng chọn tối đa để tránh quá tải Edge Server
            if len(selected_clients) >= int(len(client_importance_scores) * 0.8):
                break
    
    # Tự fall-back về min_clients nếu mạng quá tệ
    if len(selected_clients) < min_clients:
        selected_clients = [c[0] for c in sorted_clients[:min_clients]]
        selected_scores = [c[1] for c in sorted_clients[:min_clients]]
    
    # Weight = importance score chuẩn hóa
    total_importance = sum(selected_scores)
    client_weights = {}
    for i, client_id in enumerate(selected_clients):
        client_weights[client_id] = selected_scores[i] / (total_importance + 1e-10)
    
    return selected_clients, client_weights


def compute_transmission_weights(client_models_info, alpha=0.5, max_time=2.0):
    """
    Pipeline hoàn chỉnh WTO: tính importance → chọn clients → tính weights
    
    Args:
        client_models_info: list of dict, mỗi dict có:
            - 'client_id': int
            - 'class_counts': dict {class_id: count}
            - 'model_weights': state_dict
        alpha: float - tham số WTO
        bandwidth_ratio: float - giới hạn bandwidth
    
    Returns:
        selected_model_weights: list of state_dict
        aggregation_weights: list of float (cho FedWeightedAvg)
    """
    # Tính phân bố lớp toàn cục
    all_class_counts = {}
    for info in client_models_info:
        for cls, count in info['class_counts'].items():
            all_class_counts[cls] = all_class_counts.get(cls, 0) + count
    
    # Tính importance cho mỗi client
    importance_scores = {}
    for info in client_models_info:
        imp = compute_class_importance(info['class_counts'], all_class_counts, alpha)
        importance_scores[info['client_id']] = imp
    
    # Chọn clients theo WTO và Shannon-Hartley
    selected_ids, weights = wto_select_clients(importance_scores, max_time_seconds=max_time)
    
    # Lấy model weights của clients được chọn
    selected_weights = []
    agg_weights = []
    for info in client_models_info:
        if info['client_id'] in selected_ids:
            selected_weights.append(info['model_weights'])
            agg_weights.append(weights[info['client_id']])
    
    return selected_weights, agg_weights
