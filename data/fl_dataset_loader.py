import os
import json
import torch
import numpy as np

# Số clients theo FL partition mới (kịch bản 100 client, data_split/100 client)
NUM_FL_CLIENTS = 100

# Phân bố class theo Task (đã tuần tự 0-33)
FL_TASK_CLASSES_SEQUENTIAL = {
    0: list(range(0, 6)),   # Task 1: 0,1,2,3,4,5
    1: list(range(6, 12)),  # Task 2: 6,7,8,9,10,11
    2: list(range(12, 18)), # Task 3: 12..17
    3: list(range(18, 24)), # Task 4: 18..23
    4: list(range(24, 29)), # Task 5 (5 classes): 24..28
    5: list(range(29, 34)), # Task 6 (5 classes): 29..33
}

# Biến tạm để tránh lỗi Import trên Kaggle nếu main.py hoặc file khác vẫn gọi
FL_TASK_CLASSES = {}
GLOBAL_LABEL_MAP = {}

# Số mẫu tối đa mỗi lớp trong tập Test (Đặt rất lớn để tắt downsampling)
MAX_VAL_SAMPLES_PER_CLASS = 100000000

# ──────────────────────────────────────────────────────────────────────
# Remap label: data 100-client giữ NGUYÊN label ID gốc (preserve_original_label_ids)
# với thứ tự task phi tuần tự (task_mapping_label_ids.json).
# Code CIL yêu cầu label tuần tự 0..33 theo thứ tự task -> build LUT remap.
# Data cũ (đã tuần tự sẵn) không có file json này -> LUT = None, giữ nguyên label.
# ──────────────────────────────────────────────────────────────────────
_LABEL_LUT_CACHE = {}

def _get_label_lut(data_dir):
    """Trả về LUT (tensor) map label gốc -> label tuần tự, hoặc None nếu data đã tuần tự."""
    if data_dir in _LABEL_LUT_CACHE:
        return _LABEL_LUT_CACHE[data_dir]

    map_file = os.path.join(data_dir, "task_mapping_label_ids.json")
    lut = None
    if os.path.exists(map_file):
        with open(map_file, "r") as f:
            task_orders = json.load(f)  # list[list[int]]: label gốc theo từng task
        flat = [c for task in task_orders for c in task]
        lut = torch.full((max(flat) + 1,), -1, dtype=torch.long)
        for seq_id, orig_id in enumerate(flat):
            lut[orig_id] = seq_id
        print(f"[FL LOADER] Remap label gốc -> tuần tự theo: {map_file}")
        print(f"[FL LOADER] Thứ tự task (label gốc): {task_orders}")

    _LABEL_LUT_CACHE[data_dir] = lut
    return lut


def _remap_labels(y, data_dir):
    """Áp dụng LUT remap cho tensor label y (no-op nếu data đã tuần tự)."""
    lut = _get_label_lut(data_dir)
    if lut is None:
        return y
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y_new = lut[y.long()]
    if (y_new < 0).any():
        bad = torch.unique(y[y_new < 0]).tolist()
        raise ValueError(f"[FL LOADER] Label {bad} không có trong task_mapping_label_ids.json")
    return y_new

def load_fl_global_test(data_dir):
    """
    Load tập test toàn cục trực tiếp và thực hiện downsampling để tiết kiệm RAM.
    """
    test_file = os.path.join(data_dir, "global_test_data.pt")
    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"[FL LOADER] Không tìm thấy file test toàn cục: {test_file}\n"
            f"Đảm bảo file 'global_test_data.pt' tồn tại trong thư mục data_split của FL."
        )

    print(f"[FL LOADER] Đang tải Global Test Set từ: {test_file}")
    data = torch.load(test_file, weights_only=False)
    if isinstance(data, dict):
        X, y = data['x'], data['y']
    else:
        X, y = data

    y = _remap_labels(y, data_dir)

    # ──────────────────────────────────────────────────────────────────
    # Downsampling tập Test để chống sập RAM (14M mẫu -> ~100k mẫu)
    # ──────────────────────────────────────────────────────────────────
    unique_classes = torch.unique(y)
    indices_to_keep = []
    
    for cls in unique_classes:
        cls_indices = (y == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) > MAX_VAL_SAMPLES_PER_CLASS:
            # Chọn ngẫu nhiên 3000 mẫu
            perm = torch.randperm(len(cls_indices))[:MAX_VAL_SAMPLES_PER_CLASS]
            indices_to_keep.append(cls_indices[perm])
        else:
            indices_to_keep.append(cls_indices)
    
    final_indices = torch.cat(indices_to_keep)
    X = X[final_indices]
    y = y[final_indices]
    
    print(f"[FL LOADER] Global Test (Optimized): {len(y):,} mẫu | Labels: {unique_classes.tolist()}")
    return X, y


def load_fl_client_task(data_dir, task_id, client_id):
    """
    Load file train .pt cho 1 client cụ thể (labels tuần tự).
    """
    task_num = task_id + 1
    filename = f"client_{client_id}_task_{task_num}.pt"
    client_file = os.path.join(data_dir, "federated_data", filename)

    if not os.path.exists(client_file):
        return None, None

    data = torch.load(client_file, weights_only=False)
    if isinstance(data, dict):
        X, y = data['x'], data['y']
    else:
        X, y = data

    y = _remap_labels(y, data_dir)

    return X, y


def update_clients_for_task(clients_dict, data_dir, task_id):
    """
    Cập nhật dữ liệu trong RAM của tất cả clients thành dữ liệu của Task hiện tại.
    """
    task_num = task_id + 1
    print(f"\n[FL LOADER] Đang tải dữ liệu Task {task_num} lên {len(clients_dict)} Clients...")

    for cid, client in clients_dict.items():
        X_train, y_train = load_fl_client_task(data_dir, task_id, cid)

        if X_train is None or len(X_train) == 0:
            client.train_data   = torch.zeros((0, 1), dtype=torch.float32)
            client.train_labels = torch.zeros(0, dtype=torch.long)
        else:
            client.train_data   = X_train
            client.train_labels = y_train
            unique_labels = torch.unique(y_train).tolist()
            print(f"   -> Client {cid:2d}: {len(X_train):6,} mẫu | Labels: {unique_labels}")



def count_total_train_samples(data_dir, num_clients=NUM_FL_CLIENTS,
                              num_tasks=None, use_cache=True):
    """
    Dem TONG so mau train tren tat ca client & task (client_{c}_task_{t}.pt).
    Ket qua duoc cache vao file JSON de lan chay sau khong phai doc lai ~GB du lieu.

    Returns:
        int: tong so mau train.
    """
    import json

    if num_tasks is None:
        num_tasks = len(FL_TASK_CLASSES_SEQUENTIAL)

    fed_dir = os.path.join(data_dir, "federated_data")
    cache_file = os.path.join(fed_dir, "_train_sample_count.json")

    # 1) Doc cache neu khop cau hinh
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                meta = json.load(f)
            if meta.get("num_clients") == num_clients and meta.get("num_tasks") == num_tasks:
                print(f"[FL LOADER] Tong so mau train (cache): {meta['total']:,}")
                return int(meta["total"])
        except Exception:
            pass  # cache hong -> tinh lai

    # 2) Tinh bang cach load tung file, chi giu len(y)
    print(f"[FL LOADER] Dem tong so mau train tren {num_clients} clients x {num_tasks} tasks...")
    total = 0
    for cid in range(num_clients):
        for tid in range(num_tasks):
            X, y = load_fl_client_task(data_dir, tid, cid)
            if y is not None:
                total += int(len(y))
            del X, y  # giai phong RAM ngay

    print(f"[FL LOADER] Tong so mau train: {total:,}")

    # 3) Ghi cache
    if use_cache:
        try:
            os.makedirs(fed_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"total": total, "num_clients": num_clients,
                           "num_tasks": num_tasks}, f)
        except Exception:
            pass

    return total


from typing import Dict, List
from collections import defaultdict

def assign_clients_to_edges(num_clients: int, num_edge_servers: int) -> Dict[int, List[int]]:
    edge_client_map = defaultdict(list)
    clients_per_edge = num_clients // num_edge_servers
    for eid in range(num_edge_servers):
        start = eid * clients_per_edge
        end = (eid + 1) * clients_per_edge if eid < num_edge_servers - 1 else num_clients
        edge_client_map[eid] = list(range(start, end))
    return edge_client_map
