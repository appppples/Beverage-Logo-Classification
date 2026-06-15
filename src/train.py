"""
训练模块
负责模型训练、验证、保存最佳权重

训练技巧：
- Label Smoothing（标签平滑）：防止模型过度自信
- CosineAnnealing 学习率调度：比 StepLR 更平滑的衰减
- Mixup 数据增强：在样本间插值，扩充有效训练数据
"""

import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, MODEL_DIR, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING, MIXUP_ALPHA,
)


# ============================================================
# Mixup 数据增强
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """
    对一个 batch 的数据进行 Mixup 混合

    原理：随机取两张图片按比例混合
        mixed_x = λ * x_i + (1-λ) * x_j
        λ ~ Beta(alpha, alpha)

    参数:
        x: 输入图片 batch
        y: 标签 batch
        alpha: Beta 分布参数（越大混合越均匀，0.2 是常用值）

    返回:
        mixed_x: 混合后的图片
        y_a, y_b: 两组原始标签
        lam: 混合比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 对应的损失函数：按比例混合两个标签的损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 训练核心函数
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    use_mixup=False, mixup_alpha=0.2):
    """
    训练一个 epoch

    流程: 遍历数据 → (可选 Mixup) → 前向传播 → 计算损失 → 反向传播 → 更新参数

    返回:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 更新）

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        # 将数据移到 GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        if use_mixup and mixup_alpha > 0:
            # Mixup 增强
            mixed_images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha
            )
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            # Mixup 模式下，用原始标签（未混合）来计算准确率参考值
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item()
                        + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            # 正常训练
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        # 反向传播 + 参数更新
        optimizer.zero_grad()   # 清空梯度
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数

        # 统计
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    在验证集/测试集上评估模型

    与训练的区别：
    - model.eval() 模式（关闭 Dropout，BatchNorm 使用全局统计量）
    - torch.no_grad() 不计算梯度（节省显存和时间）

    返回:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# 完整训练流程
# ============================================================

def train_model(model, train_loader, val_loader, model_name: str = "model",
                use_mixup: bool = True):
    """
    完整的模型训练流程

    包含：
    - 多轮训练与验证
    - Label Smoothing 标签平滑
    - CosineAnnealing 学习率调度
    - Mixup 数据增强（可选）
    - 早停机制（Early Stopping）
    - 保存最佳模型权重

    参数:
        model: PyTorch 模型
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        model_name: 模型名称（用于保存文件命名）
        use_mixup: 是否启用 Mixup 数据增强

    返回:
        history: 训练历史记录字典
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始训练: {model_name}")
    print(f"   设备: {DEVICE}")
    print(f"   学习率: {LEARNING_RATE}")
    print(f"   总轮数: {NUM_EPOCHS}")
    print(f"   早停耐心值: {EARLY_STOP_PATIENCE}")
    print(f"   标签平滑: {LABEL_SMOOTHING}")
    print(f"   学习率调度: CosineAnnealing")
    print(f"   Mixup: {'开启 (α={})'.format(MIXUP_ALPHA) if use_mixup else '关闭'}")
    print(f"{'='*60}\n")

    # 将模型移到设备
    model = model.to(DEVICE)

    # 损失函数：交叉熵 + Label Smoothing（防止过拟合）
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # 优化器：Adam（自适应学习率，适合大多数场景）
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化可训练参数
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 学习率调度器：CosineAnnealing（比 StepLR 更平滑的衰减）
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # 早停变量
    best_val_acc = 0.0
    best_model_weights = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            use_mixup=use_mixup, mixup_alpha=MIXUP_ALPHA,
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        # 打印日志
        print(
            f"  Epoch [{epoch:>2}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.6f}  Time: {epoch_time:.1f}s"
        )

        # 早停判断
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"    ⬆️  新最佳验证准确率: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  ⏹️  早停触发！验证准确率已 {EARLY_STOP_PATIENCE} 轮未提升")
                break

    total_time = time.time() - start_time

    # 恢复最佳权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 保存模型
    save_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "history": history,
    }, save_path)

    print(f"\n{'='*60}")
    print(f"✅ 训练完成: {model_name}")
    print(f"   最佳验证准确率: {best_val_acc:.4f}")
    print(f"   总训练时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   模型已保存: {save_path}")
    print(f"{'='*60}")

    return history
