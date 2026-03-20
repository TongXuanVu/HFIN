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
from sklearn.model_selection import train_test_split
import pickle


# 43 NetFlow features (v2) - không bao gồm IP, port nguồn/đích, ID
NETFLOW_FEATURES = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES',
    'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES',
    'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES',
    'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
    'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE', 'SRC_FRAGMENTS', 'DST_FRAGMENTS'
]

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
TON_IOT_LABEL_MAP = {
    'Benign': 0,
    'DDoS': 1,
    'DoS': 2,
    'Backdoor': 3,
    'Injection': 4,
    'Ransomware': 5,
    'Scanning': 6,
    'XSS': 7,
    'MITM': 8,
    'Password': 9
}


def get_label_map(dataset_name):
    """Trả về mapping nhãn theo dataset"""
    if dataset_name == 'nf_unsw_nb15':
        return UNSW_NB15_LABEL_MAP
    elif dataset_name == 'nf_ton_iot':
        return TON_IOT_LABEL_MAP
    else:
        return UNSW_NB15_LABEL_MAP


def load_and_preprocess(data_path, dataset_name='nf_unsw_nb15',
                        test_size=0.2, random_state=42, 
                        max_samples=None, save_processed=True):
    """
    Load và tiền xử lý dataset NF-*-v2
    
    Args:
        data_path: Đường dẫn đến file CSV hoặc thư mục chứa CSV
        dataset_name: Tên dataset
        test_size: Tỷ lệ test set
        random_state: Random seed
        max_samples: Giới hạn số mẫu (None = lấy hết)
        save_processed: Lưu dữ liệu đã xử lý
    
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
    
    # Load CSV or Parquet
    print(f'[INFO] Đang load dataset từ {data_path}...')
    if os.path.isdir(data_path):
        # Tìm file CSV hoặc Parquet trong thư mục
        data_files = [f for f in os.listdir(data_path) if f.endswith('.csv') or f.endswith('.parquet')]
        if len(data_files) == 0:
            raise FileNotFoundError(f'Không tìm thấy file CSV hoặc Parquet trong {data_path}')
        
        # Ưu tiên Parquet nếu có
        parquet_files = [f for f in data_files if f.endswith('.parquet')]
        if parquet_files:
            data_path = os.path.join(data_path, parquet_files[0])
        else:
            data_path = os.path.join(data_path, data_files[0])
    
    if data_path.endswith('.parquet'):
        print(f'[INFO] Đọc file Parquet: {data_path}')
        df = pd.read_parquet(data_path)
    else:
        print(f'[INFO] Đọc file CSV: {data_path}')
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
    
    # Thực hiện Downsampling theo bài báo để cân bằng dữ liệu
    class_counts = df[label_col].value_counts()
    # Chọn target count là gấp đôi class ở percentil 50 hoặc chặn tại một mức nhất định
    target_count = int(class_counts.median() * 1.5)
    if target_count < 1000:
        target_count = 1000
        
    downsampled_dfs = []
    for cls, count in class_counts.items():
        cls_df = df[df[label_col] == cls]
        if count > target_count:
            cls_df = cls_df.sample(n=target_count, random_state=random_state)
        downsampled_dfs.append(cls_df)
    
    df = pd.concat(downsampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'[INFO] Downsampling applied to balance dataset (Target limit: {target_count})')
    print(f'[INFO] Phân bố nhãn SAU downsample:\n{df[label_col].value_counts()}')
    
    # Subsample toàn cục nếu vẫn còn lớn hơn max_samples
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
        print(f'[INFO] Subsample tổng thể xuống {max_samples} mẫu')
    
    # Xác định features có trong dataset
    available_features = [f for f in NETFLOW_FEATURES if f in df.columns]
    
    # Nếu không có features chuẩn, lấy tất cả cột số trừ cột nhãn
    if len(available_features) < 10:
        print(f'[WARN] Chỉ tìm thấy {len(available_features)} features chuẩn. Dùng tất cả cột số.')
        exclude_cols = [label_col, 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT']
        exclude_cols = [c for c in exclude_cols if c in df.columns]
        available_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f'[INFO] Sử dụng {len(available_features)} features')
    
    # Trích xuất features và labels
    X = df[available_features].values.astype(np.float32)
    
    # Mã hóa nhãn
    label_map = get_label_map(dataset_name)
    
    # Kiểm tra nếu nhãn đã là số
    if df[label_col].dtype in ['int64', 'int32', 'float64']:
        y = df[label_col].values.astype(np.int64)
    else:
        # Map string label sang số
        y = df[label_col].map(label_map)
        
        # Nếu có nhãn không nằm trong map, dùng LabelEncoder
        if y.isna().any():
            print(f'[WARN] Có nhãn không nằm trong label_map. Dùng LabelEncoder.')
            le = LabelEncoder()
            y = le.fit_transform(df[label_col])
            label_map = {label: idx for idx, label in enumerate(le.classes_)}
        else:
            y = y.values.astype(np.int64)
    
    # Xử lý NaN và Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Chuẩn hóa features (MinMaxScaler)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
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
