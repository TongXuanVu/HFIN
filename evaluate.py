"""
Đánh giá model HFIN: Accuracy, F1-score, Confusion Matrix, Forgetting
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, test_dataset, classes, device, batch_size=256):
    """
    Đánh giá model trên tập test
    
    Args:
        model: HFINNetwork
        test_dataset: NetFlowDataset
        classes: list hoặc [start, end] - phạm vi lớp cần test
        device: str
        batch_size: int
    
    Returns:
        dict với accuracy, f1_macro, f1_weighted, per_class_f1, y_true, y_pred
    """
    model.eval()
    model.to(device)

    test_dataset.getTestData(classes)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    all_preds = []
    all_labels = []

    for _, features, labels in test_loader:
        features = features.to(device)
        with torch.no_grad():
            outputs = model(features)
        preds = torch.max(outputs, dim=1)[1].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    results = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0) * 100,
        'y_true': y_true,
        'y_pred': y_pred,
    }

    model.train()
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    """Vẽ và lưu confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  [Eval] Confusion matrix saved: {save_path}')


def plot_accuracy_curve(task_accuracies, save_path):
    """Vẽ accuracy qua các task"""
    plt.figure(figsize=(10, 6))
    
    rounds = list(range(len(task_accuracies)))
    plt.plot(rounds, task_accuracies, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Global Round')
    plt.ylabel('Accuracy (%)')
    plt.title('HFIN - Global Accuracy Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  [Eval] Accuracy curve saved: {save_path}')


def compute_forgetting(task_accuracies_per_class):
    """
    Tính forgetting metric: accuracy drop trên lớp cũ sau khi học lớp mới
    
    Args:
        task_accuracies_per_class: dict {task_id: {class_id: accuracy}}
    
    Returns:
        avg_forgetting: float
    """
    if len(task_accuracies_per_class) <= 1:
        return 0.0

    forgetting_values = []
    task_ids = sorted(task_accuracies_per_class.keys())

    for i in range(len(task_ids) - 1):
        current_task = task_ids[i]
        # Lấy max accuracy của lớp này qua tất cả tasks trước
        for class_id in task_accuracies_per_class[current_task]:
            max_prev_acc = task_accuracies_per_class[current_task][class_id]
            # So sánh với accuracy cuối cùng
            final_task = task_ids[-1]
            if class_id in task_accuracies_per_class.get(final_task, {}):
                final_acc = task_accuracies_per_class[final_task][class_id]
                forgetting_values.append(max_prev_acc - final_acc)

    return np.mean(forgetting_values) if forgetting_values else 0.0


def print_evaluation_report(results, task_id, label_map=None):
    """In báo cáo đánh giá"""
    print(f'\n{"="*60}')
    print(f'  ĐÁNH GIÁ - Task {task_id}')
    print(f'{"="*60}')
    print(f'  Accuracy:     {results["accuracy"]:.2f}%')
    print(f'  F1 (macro):   {results["f1_macro"]:.2f}%')
    print(f'  F1 (weighted):{results["f1_weighted"]:.2f}%')

    if label_map:
        inv_map = {v: k for k, v in label_map.items()}
        print(f'\n  Per-class F1:')
        for i, f1 in enumerate(results['per_class_f1']):
            name = inv_map.get(i, f'Class {i}')
            print(f'    {name:20s}: {f1:.2f}%')

    print(f'{"="*60}\n')
