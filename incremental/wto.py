"""
WTO - Weighted Transmission Optimization
Mô tả: Thành phần tối ưu hóa truyền tải dữ liệu từ Client lên Edge Server.
- Client: Chỉ thu thập và gửi dữ liệu thô (NetFlow flows).
- WTO: Thuật toán chọn lọc Client dựa trên độ quan trọng dữ liệu (Eq. 8) và tốc độ truyền tải (Eq. 9).
"""
import numpy as np
import random

def calculate_shannon_hartley_rate(B_mhz=20, N0_dbm=-100):
    """
    Tính tốc độ truyền tải lý thuyết Shannon-Hartley (Mbps).
    Sử dụng các thông số từ bài báo HFIN (3.5GHz, 2.5 path loss exponent).
    """
    f = 3.5e9 
    d0 = 1.0 
    n_pl = 2.5
    SF = 8.0 
    
    # Khoảng cách giả định của Client IIoT
    d = random.uniform(10, 100)
    
    # Path loss model
    pl_f_d0 = 20 * np.log10(d0) + 20 * np.log10(f) - 147.55
    path_loss = pl_f_d0 + 10 * n_pl * np.log10(d / d0) + SF
    
    P_dbm = 23.0
    P_watt = 10**((P_dbm - 30)/10)
    N0_watt = 10**((N0_dbm - 30)/10)
    PL_linear = 10**(path_loss/10)
    
    signal_power = P_watt / PL_linear
    snr = signal_power / N0_watt
    
    B_hz = B_mhz * 1e6
    rate_bps = B_hz * np.log2(1 + snr)
    return rate_bps / 1e6


def compute_class_importance(client_class_counts, current_f1_scores, f1_threshold=0.95, beta=0.5):
    """
    Eq. 8: Tính mức độ quan trọng của dữ liệu tại Client.
    Ưu tiên các lớp có F1 thấp (khó nhận diện) và số lượng mẫu ít.
    """
    if not client_class_counts:
        return 0.0
    
    total_importance = 0.0
    for cls, count in client_class_counts.items():
        # F1_k_t: F1 score hiện tại của lớp k
        f1_t = current_f1_scores.get(cls, 0.5) 
        f1_t = max(f1_t, 0.01) # Tránh chia cho 0
        
        # w_k^i = F1* / (F1_t * (Num_k_i)^beta)
        w_k_i = f1_threshold / (f1_t * (count**beta))
        total_importance += w_k_i
        
    return total_importance


def bubble_sort_priority(client_ids, priorities):
    """
    Sắp xếp giảm dần bằng Bubble Sort theo yêu cầu của bài báo.
    """
    n = len(priorities)
    c_ids = list(client_ids)
    p_vals = list(priorities)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if p_vals[j] < p_vals[j + 1]:
                p_vals[j], p_vals[j + 1] = p_vals[j + 1], p_vals[j]
                c_ids[j], c_ids[j + 1] = c_ids[j + 1], c_ids[j]
    return c_ids, p_vals


def _wto_select_core(client_importance_scores, max_selection_ratio=0.8):
    """
    WTO logic core: Priority = Importance * Transmission Rate.
    """
    if not client_importance_scores:
        return []
    
    client_ids = list(client_importance_scores.keys())
    priorities = []
    
    for cid in client_ids:
        imp = client_importance_scores[cid]
        rate = calculate_shannon_hartley_rate()
        priorities.append(imp * rate)
    
    sorted_ids, _ = bubble_sort_priority(client_ids, priorities)
    
    num_m = max(1, int(len(client_ids) * max_selection_ratio))
    return sorted_ids[:num_m]


def wto_select_clients_for_data(all_clients_info, current_f1_scores, beta=0.5, max_selection_ratio=0.8):
    """
    Entry point cho Edge Server.
    all_clients_info: list[dict] chứa 'client_id' và 'class_counts'.
    """
    importance_scores = {}
    for info in all_clients_info:
        cid = info['client_id']
        counts = info['class_counts']
        importance_scores[cid] = compute_class_importance(counts, current_f1_scores, beta=beta)
    
    return _wto_select_core(importance_scores, max_selection_ratio=max_selection_ratio)
