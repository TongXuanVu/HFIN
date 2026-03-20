"""
Knowledge Distillation Loss cho Class-Incremental Learning
Giúp model không quên kiến thức cũ khi học lớp mới
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_one_hot(target, num_classes, device):
    """Chuyển label thành one-hot vector"""
    one_hot = torch.zeros(target.shape[0], num_classes).to(device)
    one_hot.scatter_(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def distillation_loss(outputs, old_outputs, targets, num_classes, 
                      old_num_classes, device, temperature=2.0, alpha=0.5):
    """
    Tính loss kết hợp: classification + knowledge distillation
    
    Args:
        outputs: logits từ model hiện tại (batch, num_classes)
        old_outputs: logits từ model cũ (batch, old_num_classes) hoặc None
        targets: nhãn thật (batch,)
        num_classes: tổng số lớp hiện tại
        old_num_classes: số lớp của model cũ
        device: str
        temperature: temperature cho soft targets
        alpha: trọng số distillation (0 = chỉ CE, 1 = chỉ distill)
    
    Returns:
        loss tổng hợp
    """
    # Classification loss (cross-entropy)
    target_one_hot = get_one_hot(targets, num_classes, device)
    loss_ce = F.binary_cross_entropy_with_logits(outputs, target_one_hot)
    
    if old_outputs is None or old_num_classes == 0:
        return loss_ce
    
    # Knowledge distillation loss
    # Soft targets từ model cũ
    soft_target = torch.sigmoid(old_outputs / temperature)
    soft_output = torch.sigmoid(outputs[:, :old_num_classes] / temperature)
    
    loss_kd = F.binary_cross_entropy(soft_output, soft_target.detach())
    
    # Kết hợp
    total_loss = (1 - alpha) * loss_ce + alpha * loss_kd
    
    return total_loss


def efficient_old_class_weight(output, label, num_classes, learned_classes, device):
    """
    Tính trọng số adaptive cho lớp cũ vs lớp mới
    Tương tự hàm trong GLFC gốc, đảm bảo cân bằng gradient
    
    Args:
        output: logits (batch, num_classes)
        label: nhãn (batch,)
        num_classes: tổng số lớp hiện tại
        learned_classes: list lớp đã học trước đó
        device: str
    
    Returns:
        w: trọng số (batch, 1)
    """
    pred = torch.sigmoid(output)
    N, C = pred.size(0), pred.size(1)
    
    class_mask = pred.data.new(N, C).fill_(0)
    ids = label.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    
    target = get_one_hot(label, num_classes, device)
    g = torch.abs(pred.detach() - target)
    g = (g * class_mask).sum(1).view(-1, 1)
    
    if len(learned_classes) != 0:
        ids_clone = ids.clone()
        for c in learned_classes:
            ids_clone = torch.where(ids_clone != c, ids_clone, ids_clone.clone().fill_(-1))
        
        index_old = torch.eq(ids_clone, -1).float()
        index_new = torch.ne(ids_clone, -1).float()
        
        if index_old.sum() != 0:
            w_old = torch.div(g * index_old, (g * index_old).sum() / index_old.sum())
        else:
            w_old = g.clone().fill_(0.)
        
        if index_new.sum() != 0:
            w_new = torch.div(g * index_new, (g * index_new).sum() / index_new.sum())
        else:
            w_new = g.clone().fill_(0.)
        
        w = w_old + w_new
    else:
        w = g.clone().fill_(1.)
    
    return w
