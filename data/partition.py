"""
Phân chia dữ liệu Non-IID cho Federated Learning
Mỗi client nhận một tập con lớp tấn công (mô phỏng IIoT)
"""
import numpy as np
from collections import defaultdict


def partition_data_non_iid(y_train, num_clients, num_classes_per_client=6, seed=42):
    """
    Phân chia dữ liệu non-IID: mỗi client nhận một tập con lớp
    
    Args:
        y_train: np.ndarray - nhãn train
        num_clients: int - số clients
        num_classes_per_client: int - số lớp mỗi client nhận
        seed: int - random seed
    
    Returns:
        client_data_indices: dict {client_id: np.ndarray of indices}
        client_classes: dict {client_id: list of class ids}
    """
    np.random.seed(seed)
    
    all_classes = np.unique(y_train)
    num_total_classes = len(all_classes)
    
    # Tạo index theo lớp
    class_indices = {}
    for c in all_classes:
        class_indices[c] = np.where(y_train == c)[0]
        np.random.shuffle(class_indices[c])
    
    client_data_indices = {}
    client_classes = {}
    
    for client_id in range(num_clients):
        # Chọn ngẫu nhiên một tập con lớp cho client
        selected_classes = np.random.choice(
            all_classes, 
            size=min(num_classes_per_client, num_total_classes), 
            replace=False
        )
        client_classes[client_id] = selected_classes.tolist()
        
        # Lấy dữ liệu từ các lớp đã chọn
        indices = []
        for c in selected_classes:
            c_indices = class_indices[c]
            # Mỗi client lấy một phần dữ liệu (không lấy hết)
            n_samples = len(c_indices) // num_clients
            start = (client_id * n_samples) % len(c_indices)
            end = start + n_samples
            if end <= len(c_indices):
                indices.extend(c_indices[start:end].tolist())
            else:
                indices.extend(c_indices[start:].tolist())
                indices.extend(c_indices[:end - len(c_indices)].tolist())
        
        client_data_indices[client_id] = np.array(indices)
    
    return client_data_indices, client_classes


def partition_data_by_task(y_train, task_classes, num_clients, 
                           num_classes_per_client=None, seed=42):
    """
    Phân chia dữ liệu theo task incremental cho FL
    Chỉ phân chia dữ liệu thuộc các lớp trong task hiện tại
    
    Args:
        y_train: np.ndarray - nhãn train
        task_classes: list[int] - danh sách lớp trong task hiện tại
        num_clients: int - số clients cho task này
        num_classes_per_client: int - số lớp mỗi client (None = tất cả lớp trong task)
        seed: int
    
    Returns:
        client_data_indices: dict {client_id: np.ndarray}
        client_classes: dict {client_id: list}
    """
    np.random.seed(seed)
    
    if num_classes_per_client is None:
        num_classes_per_client = len(task_classes)
    
    # Lọc chỉ lấy indices thuộc task_classes
    task_mask = np.isin(y_train, task_classes)
    task_indices = np.where(task_mask)[0]
    
    # Index theo lớp trong task
    class_indices = {}
    for c in task_classes:
        c_mask = y_train[task_indices] == c
        class_indices[c] = task_indices[c_mask]
        np.random.shuffle(class_indices[c])
    
    client_data_indices = {}
    client_classes = {}
    
    for client_id in range(num_clients):
        # Chọn lớp cho client
        n_classes = min(num_classes_per_client, len(task_classes))
        selected_classes = np.random.choice(task_classes, size=n_classes, replace=False)
        client_classes[client_id] = selected_classes.tolist()
        
        # Phân chia dữ liệu
        indices = []
        for c in selected_classes:
            c_indices = class_indices[c]
            n_per_client = max(1, len(c_indices) // num_clients)
            start = (client_id * n_per_client) % len(c_indices)
            end = min(start + n_per_client, len(c_indices))
            indices.extend(c_indices[start:end].tolist())
        
        client_data_indices[client_id] = np.array(indices)
    
    return client_data_indices, client_classes


def assign_clients_to_edges(num_clients, num_edge_servers):
    """
    Phân chia clients vào các edge servers
    
    Args:
        num_clients: int - tổng số clients
        num_edge_servers: int - số edge servers
    
    Returns:
        edge_client_map: dict {edge_id: list of client_ids}
    """
    edge_client_map = defaultdict(list)
    
    for client_id in range(num_clients):
        edge_id = client_id % num_edge_servers
        edge_client_map[edge_id].append(client_id)
    
    return dict(edge_client_map)
