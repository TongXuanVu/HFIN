"""
Cấu hình và tham số cho HFIN - Hierarchical Federated Incremental Learning NID
Thông số được căn chỉnh theo Mục VI.B của bài báo.
"""
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser(
        description='HFIN - Hierarchical Federated Class-Incremental Learning for NID'
    )

    # === Dataset ===
    parser.add_argument('--dataset', type=str, default='nf_ton_iot',
                        choices=['nf_ton_iot', 'nf_uq_nids', 'nf_unsw_nb15'],
                        help='Tên dataset (mặc định: nf_ton_iot)')
    parser.add_argument('--data_path', type=str, default='./data/raw/',
                        help='Đường dẫn thư mục chứa file CSV dataset')
    parser.add_argument('--num_features', type=int, default=43,
                        help='Số features NetFlow (43 features chuẩn)')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='Giới hạn mẫu (0 = lấy hết). Mặc định 100k để tránh treo máy với UQ-NIDS')
    parser.add_argument('--test_size', type=float, default=0.4,
                        help='Tỷ lệ test set (0.4 = 60/40 theo Mục VI.B)')

    # === Task Schedule (Class-Incremental Learning) ===
    # Tham khảo Table I trong bài báo
    parser.add_argument('--task_schedule', type=str, default='task2',
                        choices=['task2', 'task4', 'task5', 'task10'],
                        help=(
                            'Cấu hình task incremental:\n'
                            '  nf_ton_iot : task2 (5 tasks, mỗi task +2 class)\n'
                            '               task5 (2 tasks, mỗi task +5 class)\n'
                            '  nf_uq_nids : task2 (11 tasks, mỗi task +2 class)\n'
                            '               task4 ( 6 tasks, mỗi task +4 class)\n'
                            '               task10( 3 tasks, mỗi task +10 class)'
                        ))
    parser.add_argument('--num_base_classes', type=int, default=4,
                        help='Số lớp base task (tự động ghi đè từ task_schedule nếu 0)')
    parser.add_argument('--task_size', type=int, default=2,
                        help='Số lớp mới mỗi incremental task (ghi đè từ schedule)')
    parser.add_argument('--total_classes', type=int, default=10,
                        help='Tổng số class: nf_ton_iot=10, nf_uq_nids=21')
    parser.add_argument('--memory_size', type=int, default=500,
                        help='Kích thước bộ nhớ exemplar tại Edge Server (Paper: 500)')

    # === Federated Learning ===
    parser.add_argument('--num_clients', type=int, default=60,
                        help='Tổng số clients (Mục VI.B: 60 clients)')
    parser.add_argument('--num_edge_servers', type=int, default=3,
                        help='Số edge servers (Mục VI.B: 3 edges, mỗi edge 20 clients)')
    parser.add_argument('--local_clients', type=int, default=10,
                        help='Số clients được chọn mỗi vòng (mặc định: 10)')
    parser.add_argument('--epochs_base', type=int, default=40,
                        help='Số global rounds cho Base Task (Mục VI.B: 40 rounds)')
    parser.add_argument('--epochs_incremental', type=int, default=80,
                        help='Số global rounds cho mỗi Incremental Task (Mục VI.B: 80 rounds/task)')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Tham số Dirichlet non-IID (nhỏ → non-IID mạnh hơn)')
    parser.add_argument('--alpha_benign', type=float, default=0.3,
                        help='Alpha cho lớp Benign (0.3 theo Mục VI.B)')
    parser.add_argument('--alpha_attack', type=float, default=0.8,
                        help='Alpha cho các lớp Attack (0.8 theo Mục VI.B)')

    # === Huấn luyện ===
    parser.add_argument('--epochs_local', type=int, default=5,
                        help='Số epochs huấn luyện local mỗi round')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate base task (Mục VI.B: 1e-2, user đề xuất: 1e-3)')
    parser.add_argument('--lr_incremental', type=float, default=0.001,
                        help='Learning rate incremental task (Mục VI.B: 2e-2, user đề xuất: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (Mục VI.B: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD Momentum (Mục VI.B: 0.9)')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='Giảm LR sau mỗi N global rounds')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                        help='Hệ số giảm LR')

    # === WTO - Weighted Transmission Optimization ===
    parser.add_argument('--wto_beta', type=float, default=0.5,
                        help='Beta trong WTO (Eq. 8): cân bằng class importance')
    parser.add_argument('--max_transmission_time', type=float, default=2.0,
                        help='Giới hạn thời gian truyền tải WTO (giây)')

    # === Knowledge Distillation ===
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature distillation (T=2, Mục IV.B)')

    # === Model ===
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Chiều output của CNN feature extractor')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # === Khác ===
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Số global rounds giữa mỗi lần đánh giá')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help='Thư mục lưu log')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Thư mục lưu model checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Chế độ debug: dùng max_samples=50000 và epochs_global=2')

    args = parser.parse_args()

    # === Auto-fill từ dataset & task_schedule ===
    _fill_dataset_defaults(args)

    # Auto detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Debug mode: giảm số mẫu và vòng lặp
    if args.debug:
        if args.max_samples == 0:
            args.max_samples = 50000
        args.epochs_base       = 2
        args.epochs_incremental = 2
        args.num_clients    = 6
        args.num_edge_servers = 2
        args.local_clients  = 2
        args.epochs_local   = 1

    return args


def _fill_dataset_defaults(args):
    """Tự động điền total_classes, num_base_classes, task_size từ dataset & schedule."""
    from data.partition import TASK_CONFIGS

    if args.dataset not in TASK_CONFIGS:
        return

    cfg   = TASK_CONFIGS[args.dataset]
    key   = args.task_schedule

    # Lấy schedule khả dụng
    avail = list(cfg['schedules'].keys())
    if key not in avail:
        key = cfg['default_schedule']
        print(f'[CONFIG] task_schedule "{args.task_schedule}" không khả dụng với dataset '
              f'"{args.dataset}". Dùng "{key}" thay thế. Khả dụng: {avail}')
        args.task_schedule = key

    sched = cfg['schedules'][key]
    args.total_classes    = cfg['total_classes']
    args.num_base_classes = sched['base']
    args.task_size        = sched['step']

    print(f'[CONFIG] Dataset: {args.dataset} | Schedule: {key}')
    print(f'         Total classes: {args.total_classes} | '
          f'Base: {args.num_base_classes} | '
          f'Step: {args.task_size} class/task | '
          f'Tasks: {sched["num_tasks"]} incremental tasks')
