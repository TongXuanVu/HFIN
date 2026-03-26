"""
HFIN - Main Training Pipeline
Hierarchical Federated Class-Incremental Learning for Network Intrusion Detection

Luồng huấn luyện:
1. Load & tiền xử lý dataset
2. Khởi tạo: Cloud Server, Edge Servers, Clients
3. Với mỗi global round:
   a. Phân phối global model → Cloud → Edge
   b. Edge Servers thu thập dữ liệu từ Clients và huấn luyện local (WTO)
   c. Edge Servers tổng hợp weights nội bộ rồi gửi lên Cloud
   d. Cloud Server tổng hợp global (FedAvg)
   e. Đánh giá global model
"""
import os
import sys
import copy
import random
import numpy as np
import torch
import logging
from datetime import datetime

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import args_parser
from data.preprocessing import load_and_preprocess
from data.dataset import NetFlowDataset
from data.partition import partition_data_non_iid, assign_clients_to_edges
from models.feature_extractor import CNN1DFeatureExtractor, LeNetTabular, weights_init
from models.network import HFINNetwork
from federated.client import HFINClient
from federated.edge_server import EdgeServer
from federated.cloud_server import CloudServer
from federated.fed_utils import (
    setup_seed, model_to_device, FedAvg,
    model_global_eval, get_task_classes, get_all_learned_classes
)
from evaluate import evaluate_model, plot_accuracy_curve, plot_confusion_matrix, print_evaluation_report


def setup_logging(log_dir, args):
    """Thiết lập logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Thiết lập logic logging chứa cả ngày và giờ
    run_folder = datetime.now().strftime("%d-%m-%y_%H-%M")
    
    # Tạo thư mục theo cấu trúc logs/dd-mm-yy_HH-MM/
    run_log_dir = os.path.join(args.log_dir, run_folder)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Cập nhật args.log_dir để các hàm sau (như evaluate plot) tự động lưu chung vào folder này
    args.log_dir = run_log_dir
    
    log_file = os.path.join(run_log_dir, "training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('HFIN')


def main():
    # === Parse arguments ===
    args = args_parser()
    
    # === Setup ===
    setup_seed(args.seed)
    logger = setup_logging(args.log_dir, args)
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info('='*60)
    logger.info('HFIN - Hierarchical Federated Class-Incremental Learning')
    logger.info('   Network Intrusion Detection for IIoT')
    logger.info('='*60)
    logger.info(f'Device: {args.device}')
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Clients: {args.num_clients}, Edge Servers: {args.num_edge_servers}')
    logger.info(f'Base classes: {args.num_base_classes}, Task size: {args.task_size}')
    logger.info(f'Total classes: {args.total_classes}')
    
    # === Load & Tiền xử lý dữ liệu ===
    logger.info('\n[1/4] Loading dataset...')
    max_samples = 50000 if args.debug else args.max_samples
    X_train, X_test, y_train, y_test, scaler, label_map = load_and_preprocess(
        args.data_path, args.dataset, max_samples=max_samples
    )
    num_features = X_train.shape[1]
    args.total_classes = len(label_map)
    logger.info(f'Features: {num_features}, Train: {len(X_train)}, Test: {len(X_test)}')
    logger.info(f'Label map: {label_map}')
    logger.info(f'Updated total_classes to {args.total_classes} to match actual dataset!')
    
    # Test dataset (dùng chung cho đánh giá)
    test_dataset = NetFlowDataset(X_test, y_test)
    
    # === Phân chia dữ liệu cho clients ===
    logger.info('\n[2/4] Partitioning data to clients...')
    # Phân phối Dirichlet theo Mục VI.B: Alpha = 0.3 (Benign: 0), Alpha = 0.8 (Attack: >0)
    alpha_config = {0: args.alpha_benign} # Lớp 0 (Benign) dùng alpha_benign=0.3
    # Các lớp khác tự động dùng alpha_attack=0.8 trong logic partition_data_non_iid
    
    client_data_indices, client_classes = partition_data_non_iid(
        y_train, args.num_clients, alpha=alpha_config, seed=args.seed
    )
    
    # Gán clients vào edge servers
    edge_client_map = assign_clients_to_edges(args.num_clients, args.num_edge_servers)
    logger.info(f'Edge-Client mapping: {edge_client_map}')
    
    # === Khởi tạo models & components ===
    logger.info('\n[3/4] Initializing models...')
    feature_extractor = CNN1DFeatureExtractor(
        input_dim=num_features,
        output_dim=args.feature_dim
    )
    
    # Encode model (cho prototype gradient)
    encode_model = LeNetTabular(
        input_dim=num_features,
        hidden_dim=128,
        num_classes=args.total_classes
    )
    encode_model.apply(weights_init)
    
    # Global model
    model_g = HFINNetwork(args.num_base_classes, copy.deepcopy(feature_extractor))
    model_g = model_to_device(model_g, args.device)
    
    # Tạo clients (Data Providers)
    clients_dict = {}
    for c_id in range(args.num_clients):
        indices = client_data_indices[c_id]
        client = HFINClient(
            client_id=c_id,
            train_data=X_train[indices],
            train_labels=y_train[indices],
            device=args.device
        )
        clients_dict[c_id] = client
    
    # Tạo edge servers (Local Trainers)
    edge_servers = []
    for e_id in range(args.num_edge_servers):
        edge = EdgeServer(
            edge_id=e_id, 
            num_classes=args.num_base_classes,
            feature_extractor=copy.deepcopy(feature_extractor),
            device=args.device,
            memory_size=args.memory_size,
            task_size=args.task_size
        )
        edge.set_clients(edge_client_map[e_id])
        edge_servers.append(edge)
    
    # === Cloud server ===
    cloud_server = CloudServer(
        num_classes=args.num_base_classes,
        feature_extractor=copy.deepcopy(feature_extractor),
        device=args.device,
        learning_rate=args.learning_rate,
        encode_model=copy.deepcopy(encode_model)
    )
    
    # === Training Loop (Task-based) ===
    logger.info('\n[4/4] Starting training...')
    accuracy_history = []
    task_accuracies_per_class = {}  # Lưu accuracy từng lớp theo round để tính forgetting
    classes_learned = args.num_base_classes
    current_f1_scores = {i: 0.9 for i in range(args.total_classes)}
    global_round = 0  # Bộ đếm round tổng (để log)

    # Xây dựng danh sách tasks từ task_schedule
    # Mỗi task là một list class IDs, ví dụ: [[0,1], [2,3], [4,5], [6,7], [8,9]]
    from data.partition import get_task_schedule
    all_task_classes = get_task_schedule(args.dataset, args.task_schedule)
    num_tasks = len(all_task_classes)

    for task_id, task_classes in enumerate(all_task_classes):
        # Số rounds cho task này (Sec VI.B)
        num_rounds = args.epochs_base if task_id == 0 else args.epochs_incremental

        # === Chuẩn bị model cho task mới ===
        # WA: Weight Aligning (Eq. 9) trước khi mở rộng
        if task_id > 0:
            model_g.weight_align(classes_learned - args.task_size, classes_learned)

        # Cập nhật số lớp đã học
        classes_learned = len(task_classes) if task_id == 0 else classes_learned + len(task_classes)

        # Mở rộng Classification Head nếu cần
        if classes_learned > model_g.fc.out_features:
            model_g.Incremental_learning(classes_learned)
            logger.info(f'Expanded global model to {classes_learned} classes for Task {task_id}')

        # Đồng bộ kiến trúc sang Cloud & Edge
        model_g = model_to_device(model_g, args.device)
        cloud_server.model = copy.deepcopy(model_g)
        for edge in edge_servers:
            edge.model = copy.deepcopy(model_g)
            edge.learned_classes = list(range(classes_learned))

        # Learning Rate theo Sec VI.B
        current_lr = args.learning_rate if task_id == 0 else args.lr_incremental
        logger.info(f'\n{"="*60}')
        logger.info(f'Task {task_id}/{num_tasks - 1} | Classes: {task_classes} | '
                    f'{num_rounds} rounds | LR: {current_lr}')
        logger.info(f'{"="*60}')

        # === Inner loop: rounds cho task này ===
        for round_in_task in range(num_rounds):
            global_round += 1

            logger.info(f'\n--- Task {task_id}, Round {round_in_task + 1}/{num_rounds} '
                        f'(Global {global_round}) ---')

            # Phân phối Global Model về các Edges
            global_weights = cloud_server.get_weights()
            for edge in edge_servers:
                edge.set_weights(global_weights)

            # === Edge Training (WTO + Data Collection + FCIL) ===
            edge_weights = []
            edge_samples = []

            for edge in edge_servers:
                weights, num_samples = edge.train_local(
                    clients_dict=clients_dict,
                    global_round=global_round,
                    task_id=task_id,
                    task_classes=task_classes,
                    current_f1_scores=current_f1_scores,
                    epochs=args.epochs_local,
                    lr=current_lr,
                    batch_size=args.batch_size
                )
                if weights:
                    edge_weights.append(weights)
                    edge_samples.append(num_samples)

            # === Cloud-level Global Aggregation ===
            if edge_weights:
                global_weights_new = cloud_server.aggregate_from_edges(edge_weights, edge_samples)
                model_g.load_state_dict(global_weights_new)

            # Cập nhật cloud server state
            cloud_server.model.load_state_dict(model_g.state_dict())

            # === Đánh giá Định kỳ (Sec VI.B: every 5 rounds) ===
            is_first_round = (round_in_task == 0 and task_id == 0)
            is_eval_round = (round_in_task + 1) % args.eval_interval == 0
            is_last_round_of_task = (round_in_task == num_rounds - 1)

            if is_first_round or is_eval_round or is_last_round_of_task:
                acc = model_global_eval(
                    model_g, test_dataset, task_id, args.task_size,
                    args.num_base_classes, args.device
                )
                accuracy_history.append(acc)
                logger.info(f'  [Eval] Task {task_id}, Round {round_in_task + 1}: '
                            f'Accuracy = {acc:.2f}%')

                # Lưu accuracy phục vụ tính Forgetting
                results = evaluate_model(model_g, test_dataset, range(classes_learned), args.device)
                task_accuracies_per_class[global_round] = {
                    i: f1 for i, f1 in enumerate(results['per_class_f1'])
                }

    # === Kết thúc: Đánh giá cuối cùng & Forgetting Metric ===
    from evaluate import compute_forgetting
    avg_forgetting = compute_forgetting(task_accuracies_per_class)
    logger.info(f'\n{"="*60}')
    logger.info(f'  KẾT QUẢ CUỐI CÙNG')
    logger.info(f'  Final Accuracy: {accuracy_history[-1]:.2f}%')
    logger.info(f'  Avg Forgetting: {avg_forgetting:.2f}%')
    logger.info(f'{"="*60}')
    logger.info('\n' + '='*60)
    logger.info('TRAINING COMPLETED!')
    logger.info('='*60)
    
    # Đánh giá chi tiết
    final_classes = list(range(min(classes_learned, args.total_classes)))
    results = evaluate_model(model_g, test_dataset, final_classes, args.device)
    print_evaluation_report(results, task_id, label_map)
    
    # Vẽ biểu đồ
    plot_accuracy_curve(accuracy_history, os.path.join(args.log_dir, 'accuracy_curve.png'))
    
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map.get(i, f'Class {i}') for i in final_classes]
    plot_confusion_matrix(
        results['y_true'], results['y_pred'], class_names,
        os.path.join(args.log_dir, 'confusion_matrix.png'),
        title=f'HFIN Confusion Matrix - Task {task_id}'
    )
    
    # Lưu model
    save_path = os.path.join(args.save_dir, 'hfin_final_model.pth')
    torch.save({
        'model_state_dict': model_g.state_dict(),
        'args': vars(args),
        'accuracy_history': accuracy_history,
        'final_results': {
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted']
        }
    }, save_path)
    logger.info(f'Model saved: {save_path}')


if __name__ == '__main__':
    main()
