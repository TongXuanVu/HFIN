"""
HFIN - Main Training Pipeline
Hierarchical Federated Class-Incremental Learning for Network Intrusion Detection

Luồng huấn luyện:
1. Load & tiền xử lý dataset
2. Khởi tạo: Cloud Server, Edge Servers, Clients
3. Với mỗi global round:
   a. Phân phối global model → edge → clients  
   b. Clients huấn luyện local
   c. Edge servers tổng hợp (WTO)
   d. Cloud server tổng hợp global
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
from models.feature_extractor import MLPFeatureExtractor, LeNetTabular, weights_init
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
    client_data_indices, client_classes = partition_data_non_iid(
        y_train, args.num_clients, seed=args.seed
    )
    
    # Gán clients vào edge servers
    edge_client_map = assign_clients_to_edges(args.num_clients, args.num_edge_servers)
    logger.info(f'Edge-Client mapping: {edge_client_map}')
    
    # === Khởi tạo models & components ===
    logger.info('\n[3/4] Initializing models...')
    feature_extractor = MLPFeatureExtractor(
        input_dim=num_features,
        hidden_dims=args.hidden_dims,
        output_dim=args.feature_dim,
        dropout=args.dropout
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
    
    # Tạo clients
    clients = []
    for c_id in range(args.num_clients):
        indices = client_data_indices[c_id]
        client = HFINClient(
            client_id=c_id,
            num_classes=args.num_base_classes,
            feature_extractor=copy.deepcopy(feature_extractor),
            batch_size=args.batch_size,
            task_size=args.task_size,
            memory_size=args.memory_size,
            epochs=args.epochs_local,
            learning_rate=args.learning_rate,
            train_data=X_train[indices],
            train_labels=y_train[indices],
            device=args.device,
            num_base_classes=args.num_base_classes,
            encode_model=copy.deepcopy(encode_model)
        )
        clients.append(client)
    
    # Tạo edge servers
    edge_servers = []
    for e_id in range(args.num_edge_servers):
        edge = EdgeServer(edge_id=e_id, device=args.device)
        edge.set_clients(edge_client_map[e_id])
        edge_servers.append(edge)
    
    # Tạo cloud server
    cloud_server = CloudServer(
        num_classes=args.num_base_classes,
        feature_extractor=copy.deepcopy(feature_extractor),
        device=args.device,
        learning_rate=args.learning_rate,
        encode_model=copy.deepcopy(encode_model)
    )
    
    # === Training Loop ===
    logger.info('\n[4/4] Starting training...')
    accuracy_history = []
    old_task_id = -1
    classes_learned = args.num_base_classes
    
    # Quản lý old/new clients
    old_client_0 = []  # Clients cũ không train
    old_client_1 = list(range(args.num_clients))  # Clients cũ có train
    new_client = []  # Clients mới
    
    for ep_g in range(args.epochs_global):
        pool_grad = []
        task_id = ep_g // args.tasks_global
        
        # === Phát hiện task mới ===
        if task_id != old_task_id and old_task_id != -1:
            # Cập nhật client groups
            overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
            new_client_count = min(args.task_size, args.num_clients - overall_client)
            if new_client_count > 0:
                new_client = list(range(overall_client, overall_client + new_client_count))
            
            active_clients = list(range(min(overall_client + new_client_count, args.num_clients)))
            n_old_1 = int(len(active_clients) * 0.9)
            old_client_1 = random.sample(active_clients, n_old_1)
            old_client_0 = [i for i in active_clients if i not in old_client_1]
            
            # Mở rộng model cho lớp mới
            classes_learned += args.task_size
            classes_learned = min(classes_learned, args.total_classes)
            model_g.Incremental_learning(classes_learned)
            model_g = model_to_device(model_g, args.device)
            cloud_server.model.Incremental_learning(classes_learned)
        
        logger.info(f'\n--- Global Round {ep_g}/{args.epochs_global}, Task {task_id} ---')
        
        # === Lấy old model cho distillation ===
        model_old = cloud_server.model_back()
        
        # === Chọn clients cho round này ===
        num_active = len(old_client_0) + len(old_client_1) + len(new_client)
        num_active = min(num_active, args.num_clients)
        available_clients = list(range(num_active))
        selected_clients = random.sample(
            available_clients, min(args.local_clients, len(available_clients))
        )
        logger.info(f'  Selected clients: {selected_clients}')
        
        # === Local Training ===
        client_models_info = []
        for c_id in selected_clients:
            if c_id >= len(clients):
                continue
                
            # Cập nhật model global cho client
            clients[c_id].model = copy.deepcopy(model_g)
            
            # Chuẩn bị dữ liệu
            group = 0 if c_id in old_client_0 else 1
            clients[c_id].beforeTrain(task_id, group)
            clients[c_id].update_exemplar()
            
            # Huấn luyện local
            clients[c_id].train(ep_g, model_old)
            
            # Thu thập model weights và class info
            info = {
                'client_id': c_id,
                'model_weights': clients[c_id].get_model_weights(),
                'class_counts': clients[c_id].get_class_counts()
            }
            client_models_info.append(info)
            
            # Thu thập prototype gradients
            proto = clients[c_id].proto_grad_sharing()
            if proto is not None:
                pool_grad.extend(proto)
        
        # === Cập nhật exemplar cho clients không train ===
        for c_id in range(min(num_active, len(clients))):
            if c_id not in selected_clients:
                clients[c_id].model = copy.deepcopy(model_g)
                group = 0 if c_id in old_client_0 else 1
                clients[c_id].beforeTrain(task_id, group)
                clients[c_id].update_exemplar()
        
        # === Edge-level Aggregation (WTO) ===
        edge_weights = []
        for edge in edge_servers:
            # Lọc client models thuộc edge này
            edge_clients_info = [
                info for info in client_models_info
                if info['client_id'] in edge.client_ids
            ]
            if edge_clients_info:
                agg = edge.aggregate(
                    edge_clients_info, use_wto=True,
                    wto_alpha=args.wto_alpha,
                    max_time=2.0
                )
                if agg is not None:
                    edge_weights.append(agg)
        
        # === Cloud-level Global Aggregation ===
        if edge_weights:
            global_weights = cloud_server.aggregate_from_edges(edge_weights)
            model_g.load_state_dict(global_weights)
        
        # Cập nhật cloud server
        cloud_server.model = copy.deepcopy(model_g)
        cloud_server.update_monitor(pool_grad if pool_grad else None)
        
        # === Đánh giá ===
        acc = model_global_eval(
            model_g, test_dataset, task_id, args.task_size,
            args.num_base_classes, args.device
        )
        accuracy_history.append(acc)
        
        logger.info(f'  Task {task_id}, Round {ep_g}: Accuracy = {acc:.2f}%')
        
        old_task_id = task_id
    
    # === Kết thúc: Đánh giá cuối cùng ===
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
