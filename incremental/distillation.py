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
                      old_num_classes, device, temperature=2.0, class_weight=None):
    """
    Tính loss kết hợp theo Eq. 7 từ bài báo HFIN:
    L = lambda1 * L_CE + lambda2 * L_KD

    Trong đó:
    lambda1 = n_new / (n_old + n_new)  (Tỷ lệ lớp mới)
    lambda2 = n_old / (n_old + n_new)  (Tỷ lệ lớp cũ)

    Args:
        outputs: logits từ model hiện tại (batch, num_classes)
        old_outputs: logits từ model cũ (batch, old_num_classes) hoặc None
        targets: nhãn thật (batch,)
        num_classes: tổng số lớp hiện tại (n_old + n_new)
        old_num_classes: số lớp của model cũ (n_old)
        device: str
        temperature: temperature cho soft targets (T)
        class_weight: Tensor trọng số lớp (num_classes,) để cân bằng imbalance, hoặc None
    """
    # 1. Classification loss (cross-entropy) - dùng class_weight nếu có
    loss_ce = F.cross_entropy(outputs, targets, weight=class_weight)

    if old_outputs is None or old_num_classes == 0:
        return loss_ce

    # 2. Knowledge distillation loss (Eq. 11, 12)
    # Sử dụng Softmax + Temperature
    # p_old: soft targets từ model cũ
    # p_new: soft predictions từ model mới (chỉ lấy phần các lớp cũ)
    p_old = F.softmax(old_outputs / temperature, dim=1)
    p_new = F.log_softmax(outputs[:, :old_num_classes] / temperature, dim=1)
    
    # KL Divergence là cách chuẩn để tính KD loss với softmax
    loss_kd = F.kl_div(p_new, p_old.detach(), reduction='batchmean') * (temperature ** 2)
    
    # 3. Hệ số Lambdas (Theo bài báo HFIN - Eq. 7 thường là tổng trực tiếp hoặc lambdas=1)
    # n_old = float(old_num_classes)
    # n_new = float(num_classes - old_num_classes)
    # n_total = float(num_classes)
    
    # Người dùng xác nhận bài báo sử dụng lambda1=1.0, lambda2=1.0 (như iCaRL)
    lambda1 = 1.0  # Trọng số cho lớp mới
    lambda2 = 1.0  # Trọng số cho lớp cũ (Distillation)
    
    # Kết hợp (Eq. 7)
    total_loss = lambda1 * loss_ce + lambda2 * loss_kd
    
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
