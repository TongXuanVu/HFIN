"""
Tiền xử lý dữ liệu NF-UNSW-NB15-v2 (và các dataset NF-*-v2 khác)
- Load CSV
- Chọn 43 NetFlow features
- Chuẩn hóa features
- Mã hóa nhãn
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pickle

# Import downloader (tự động tải nếu chưa có dataset)
try:
    from data.downloader import ensure_dataset
except ImportError:
    try:
        from downloader import ensure_dataset
    except ImportError:
        ensure_dataset = None


# --- Bỏ NETFLOW_FEATURES cũ để dùng logic loại trừ (exclusion) đảm bảo 41 features ---

# Mapping nhãn tấn công cho NF-UNSW-NB15-v2
UNSW_NB15_LABEL_MAP = {
    'Benign': 0,
    'Fuzzers': 1,
    'Analysis': 2,
    'Backdoors': 3,
    'DoS': 4,
    'Exploits': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
}

# Mapping cho NF-ToN-IoT-v2
# Task 1: Benign, Scanning (Mục VI.B)
TON_IOT_LABEL_MAP = {
    'benign': 0,
    'scanning': 1,
    'ddos': 2,
    'dos': 3,
    'backdoor': 4,
    'injection': 5,
    'ransomware': 6,
    'xss': 7,
    'mitm': 8,
    'password': 9
}


# Mapping cho NF-UQ-NIDS-v2 (21 class: Benign + 20 loại tấn công)
# Task 1: Benign, DDoS (Mục VI.B)
UQ_NIDS_LABEL_MAP = {
    'benign': 0,
    'ddos': 1,
    'dos': 2,
    'backdoor': 3,
    'backdoors': 3, # Map variant to same index
    'injection': 4,
    'ransomware': 5,
    'scanning': 6,
    'xss': 7,
    'mitm': 8,
    'password': 9,
    'analysis': 10,
    'bot': 11,
    'brute force': 12,
    'exploits': 13,
    'fuzzers': 14,
    'infilteration': 15,
    'infiltration': 15, # Corrected variant
    'reconnaissance': 16,
    'shellcode': 17,
    'theft': 18,
    'worms': 19,
    'web-sql': 20,
    'web-bfa': 20, # Grouping variants or making room for 21 classes
    'web-xss': 20,
    'generic': 15 # Map generic to a placeholder if not in paper
}
# Ghi chú: Một số lớp có thể trùng lặp tên do gộp nhiều dataset, 
# nhưng Label.unique() sẽ cho ta danh sách chuẩn.


def get_label_map(dataset_name):
    """Trả về mapping nhãn theo dataset"""
    if dataset_name == 'nf_ton_iot':
        return TON_IOT_LABEL_MAP
    elif dataset_name == 'nf_uq_nids':
        return UQ_NIDS_LABEL_MAP
    else:
        return UNSW_NB15_LABEL_MAP


def load_and_preprocess(data_path, dataset_name='nf_ton_iot',
                        test_size=0.4, random_state=42,   # ← 60/40 theo bài báo
                        max_samples=None, save_processed=True):
    """
    Load và tiền xử lý dataset NF-*-v2

    Args:
        data_path    : Đường dẫn đến file CSV hoặc thư mục chứa CSV
        dataset_name : Tên dataset
        test_size    : Tỷ lệ test (0.4 = 60/40 theo Mục VI.B)
        random_state : Random seed
        max_samples  : Giới hạn số mẫu (None = lấy hết)
        save_processed: Lưu dữ liệu đã xử lý (.pkl)

    Returns:
        X_train, X_test, y_train, y_test, scaler, label_map
    """
    processed_path = os.path.join(os.path.dirname(data_path), f'{dataset_name}_processed.pkl')
    
    # Nếu đã có dữ liệu xử lý, load luôn
    if save_processed and os.path.exists(processed_path):
        print(f'[INFO] Đang load dữ liệu đã xử lý từ {processed_path}...')
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['X_test'], data['y_train'], data['y_test'], \
               data['scaler'], data['label_map']
    
    # === Tự động tải dataset nếu chưa có ===
    if ensure_dataset is not None:
        try:
            # ensure_dataset giờ trả về đường dẫn file cụ thể
            data_path = ensure_dataset(dataset_name, data_path)
        except FileNotFoundError as e:
            raise e
    
    # Load CSV or Parquet
    print(f'[INFO] Đang load dataset từ {data_path}...')
    if os.path.isdir(data_path):
        # Nếu vẫn là thư mục, tìm file khớp với dataset_name
        data_files = [f for f in os.listdir(data_path) if f.endswith('.csv') or f.endswith('.parquet')]
        if len(data_files) == 0:
            raise FileNotFoundError(f'Không tìm thấy file CSV hoặc Parquet trong {data_path}')
        
        # Ưu tiên file có tên chứa dataset_name (ví dụ: 'uq-nids' hoặc 'ton-iot')
        keyword = dataset_name.lower().replace('nf_', '').replace('_', '-')
        match_files = [f for f in data_files if keyword in f.lower()]
        
        if match_files:
            data_path = os.path.join(data_path, match_files[0])
        else:
            # Dự phòng: Ưu tiên Parquet
            parquet_files = [f for f in data_files if f.endswith('.parquet')]
            data_path = os.path.join(data_path, parquet_files[0] if parquet_files else data_files[0])
    
    if data_path.endswith('.parquet'):
        print(f'[INFO] Đọc file Parquet: {data_path}')
        df = pd.read_parquet(data_path)
    else:
        print(f'[INFO] Đọc file CSV: {data_path}')
        # Optimize: Dùng nrows nếu max_samples được cung cấp để tránh load 13GB vào RAM
        if dataset_name == 'nf_uq_nids' and (not max_samples or max_samples == 0):
            print(f'[WARN] Dataset {dataset_name} quá lớn (13.7GB). Tự động giới hạn đọc 500,000 dòng để tránh treo máy.')
            df = pd.read_csv(data_path, nrows=500000)
        elif max_samples and max_samples > 0:
            # Lấy dư ra một chút (5x) để đảm bảo có đủ mẫu sau khi downsample các lớp hiếm
            df = pd.read_csv(data_path, nrows=max_samples * 5)
        else:
            df = pd.read_csv(data_path)
    print(f'[INFO] Dataset shape: {df.shape}')
    print(f'[INFO] Columns: {list(df.columns)}')
    
    # Xác định cột nhãn
    label_col = None
    for col in ['Attack', 'attack', 'Label', 'label', 'Attack_type', 'attack_cat']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f'Không tìm thấy cột nhãn. Các cột có: {list(df.columns)}')
    
    print(f'[INFO] Cột nhãn: {label_col}')
    print(f'[INFO] Phân bố nhãn TRƯỚC downsample:\n{df[label_col].value_counts()}')
    
    # Thực hiện Downsampling theo bài báo để cân bằng dữ liệu (nhưng vẫn giữ ưu thế cho Benign)
    class_counts = df[label_col].value_counts()
    # Tăng giới hạn lên để heatmap không bị trắng xóa (Sparse) khi chia cho 60 client
    target_count = 5000 
        
    downsampled_dfs = []
    for cls, count in class_counts.items():
        cls_df = df[df[label_col] == cls]
        # Lớp Benign thường được giữ nhiều hơn để tránh False Positives
        current_target = target_count * 5 if str(cls).lower() == 'benign' else target_count
        
        if count > current_target:
            cls_df = cls_df.sample(n=current_target, random_state=random_state)
        downsampled_dfs.append(cls_df)
    
    df = pd.concat(downsampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'[INFO] Downsampling applied to balance dataset (Target limit: {target_count})')
    print(f'[INFO] Phân bố nhãn SAU downsample:\n{df[label_col].value_counts()}')
    
    # Subsample toàn cục nếu vẫn còn lớn hơn max_samples
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
        print(f'[INFO] Subsample tổng thể xuống {max_samples} mẫu')
    
    # Loại bỏ các cột không phải feature theo bài báo (để còn lại flow-based features)
    # Mục VI.A: "we exclude source and destination IP addresses, and source and destination port numbers."
    exclude_cols = [
        'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 
        'L4_SRC_PORT', 'L4_DST_PORT',
        'Label', 'Attack', 'Dataset', 'attack', 'label', 'Attack_type', 'attack_cat'
    ]
    available_features = [c for c in df.columns if c not in exclude_cols]
    
    print(f'[INFO] Sử dụng {len(available_features)} features (Đúng chuẩn 41 features của bài báo)')
    
    # Trích xuất features và labels
    X = df[available_features].values.astype(np.float32)
    
    # Mã hóa nhãn (Case-insensitive mapping)
    label_map = {k.lower(): v for k, v in get_label_map(dataset_name).items()}
    
    # Map string label sang số (chuyển về lowercase trước khi map)
    y_series = df[label_col].astype(str).str.lower().map(label_map)
    
    # Nếu có nhãn không nằm trong map, dùng LabelEncoder cho các nhãn còn lại
    if y_series.isna().any():
        print(f'[WARN] Có nhãn không nằm trong label_map. Tự động mã hóa phần còn lại.')
        # Giữ lại các nhãn đã map thành công
        mask = y_series.isna()
        known_indices = y_series.dropna().unique().astype(int)
        next_idx = max(known_indices) + 1 if len(known_indices) > 0 else 0
        
        le = LabelEncoder()
        y_series[mask] = le.fit_transform(df.loc[mask, label_col]) + next_idx
        # Cập nhật label_map
        for i, cls_name in enumerate(le.classes_):
            label_map[cls_name.lower()] = i + next_idx
            
    y = y_series.values.astype(np.int64)
    
    # Xử lý NaN và Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Kiểm tra số lượng mẫu mỗi lớp để tránh lỗi StratifiedShuffleSplit
    unique_classes, counts = np.unique(y, return_counts=True)
    rare_classes = unique_classes[counts < 2]
    if len(rare_classes) > 0:
        print(f'[WARN] Loại bỏ {len(rare_classes)} lớp có ít hơn 2 mẫu: {rare_classes}')
        mask = ~np.isin(y, rare_classes)
        X = X[mask]
        y = y[mask]
        
    # Chuẩn hóa features (MinMaxScaler)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Chia tập Train/Test (60/40 theo bài báo)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    print(f'[INFO] Train: {X_train.shape}, Test: {X_test.shape}')
    print(f'[INFO] Số lớp: {len(np.unique(y))}')
    
    # Lưu dữ liệu đã xử lý
    if save_processed:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        with open(processed_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'scaler': scaler, 'label_map': label_map,
                'feature_names': available_features
            }, f)
        print(f'[INFO] Đã lưu dữ liệu xử lý tại {processed_path}')
    
    return X_train, X_test, y_train, y_test, scaler, label_map


if __name__ == '__main__':
    # Test nhanh
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = './data/raw/'
    
    X_train, X_test, y_train, y_test, scaler, label_map = load_and_preprocess(path)
    print(f'\n=== Kết quả tiền xử lý ===')
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'Label map: {label_map}')
    print(f'Phân bố train: {np.bincount(y_train)}')
    print(f'Phân bố test: {np.bincount(y_test)}')
