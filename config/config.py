"""
Cấu hình và tham số cho HFIN - Hierarchical Federated Incremental Learning NID
"""
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser(description='HFIN - Hierarchical Federated Class-Incremental Learning for NID')

    # === Dataset ===
    parser.add_argument('--dataset', type=str, default='nf_unsw_nb15',
                        choices=['nf_unsw_nb15', 'nf_ton_iot', 'nf_uq_nids'],
                        help='Tên dataset')
    parser.add_argument('--data_path', type=str, default='./data/raw/',
                        help='Đường dẫn thư mục chứa file CSV dataset')
    parser.add_argument('--num_features', type=int, default=43,
                        help='Số features NetFlow')
    parser.add_argument('--max_samples', type=int, default=3000000,
                        help='Giới hạn số mẫu dữ liệu tối đa (để tránh tràn RAM)')

    # === Mô hình ===
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 128],
                        help='Kích thước các lớp ẩn của feature extractor MLP')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Chiều output của feature extractor')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # === Class-Incremental Learning ===
    parser.add_argument('--num_base_classes', type=int, default=4,
                        help='Số lớp trong task đầu tiên (base task)')
    parser.add_argument('--task_size', type=int, default=2,
                        help='Số lớp mới mỗi task incremental')
    parser.add_argument('--total_classes', type=int, default=10,
                        help='Tổng số lớp (Benign + 9 loại tấn công)')
    parser.add_argument('--memory_size', type=int, default=2000,
                        help='Kích thước bộ nhớ exemplar (số mẫu lưu lại)')

    # === Federated Learning ===
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Tổng số clients ban đầu')
    parser.add_argument('--num_edge_servers', type=int, default=3,
                        help='Số edge servers trong kiến trúc phân cấp')
    parser.add_argument('--local_clients', type=int, default=5,
                        help='Số clients được chọn mỗi round')
    parser.add_argument('--epochs_global', type=int, default=100,
                        help='Tổng số global rounds')
    parser.add_argument('--tasks_global', type=int, default=10,
                        help='Số global rounds cho mỗi task')

    # === Huấn luyện ===
    parser.add_argument('--epochs_local', type=int, default=20,
                        help='Số epochs huấn luyện local mỗi round')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='Giảm LR sau mỗi N global rounds')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                        help='Hệ số giảm LR')

    # === WTO - Weighted Transmission Optimization ===
    parser.add_argument('--wto_alpha', type=float, default=0.5,
                        help='Trọng số alpha trong WTO (cân bằng tần suất vs importance)')
    parser.add_argument('--bandwidth_ratio', type=float, default=0.7,
                        help='Tỷ lệ bandwidth giới hạn cho WTO (0-1)')

    # === Knowledge Distillation ===
    parser.add_argument('--distill_weight', type=float, default=0.5,
                        help='Trọng số của distillation loss')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature cho knowledge distillation')

    # === Khác ===
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, cuda:1, ...')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help='Thư mục lưu log')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Thư mục lưu model')
    parser.add_argument('--debug', action='store_true',
                        help='Chế độ debug (ít dữ liệu, ít rounds)')

    args = parser.parse_args()

    # Auto detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args
