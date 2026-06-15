"""
评估与可视化模块
在测试集上评估模型，生成各种图表和分析结果

评估技巧：
- 测试时增强（TTA）：对同一张图片做多种变换后取平均，提高预测稳定性
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_fscore_support,
)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, RESULT_DIR, CLASSES, CLASS_NAMES_CN, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 模型预测（收集所有预测结果）
# ============================================================

def get_predictions(model, dataloader, device):
    """
    在数据集上运行模型，收集所有预测结果

    返回:
        all_labels: 真实标签列表
        all_preds: 预测标签列表
        all_probs: 预测概率矩阵 (N × NUM_CLASSES)
        all_paths: 图片路径列表（如果 dataset 有 samples 属性）
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ============================================================
# 图表 1 & 2：训练曲线
# ============================================================

def plot_training_curves(history: dict, model_name: str):
    """
    绘制训练/验证的 Loss 曲线和 Accuracy 曲线

    保存两张图:
        - results/{model_name}_loss_curve.png
        - results/{model_name}_acc_curve.png
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="训练集 Loss", color="#FF6B6B", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s-", label="验证集 Loss", color="#4ECDC4", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{model_name} — 损失函数变化曲线", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"{model_name}_loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Loss 曲线已保存: {save_path}")

    # --- Accuracy 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], "o-", label="训练集 Accuracy", color="#FF6B6B", linewidth=2)
    ax.plot(epochs, history["val_acc"], "s-", label="验证集 Accuracy", color="#4ECDC4", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{model_name} — 准确率变化曲线", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"{model_name}_acc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Accuracy 曲线已保存: {save_path}")


# ============================================================
# 图表 3：混淆矩阵
# ============================================================

def plot_confusion_matrix(labels, preds, model_name: str):
    """
    绘制混淆矩阵热力图（中文标签）

    保存: results/{model_name}_confusion_matrix.png
    """
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES_CN,
        yticklabels=CLASS_NAMES_CN,
        ax=ax, annot_kws={"size": 14},
    )
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(f"{model_name} — 混淆矩阵", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🔥 混淆矩阵已保存: {save_path}")


# ============================================================
# 图表 4：分类报告
# ============================================================

def print_classification_report(labels, preds, model_name: str):
    """
    打印并保存分类报告（Precision / Recall / F1）

    保存: results/{model_name}_classification_report.txt
    """
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES_CN,
        digits=4,
    )

    print(f"\n  📊 {model_name} 分类报告:")
    print(report)

    save_path = os.path.join(RESULT_DIR, f"{model_name}_classification_report.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name} 分类报告\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  📊 分类报告已保存: {save_path}")

    return report


# ============================================================
# 图表 5：ROC 曲线 + AUC
# ============================================================

def plot_roc_curves(labels, probs, model_name: str):
    """
    绘制每个品牌的 ROC 曲线和 AUC 值

    使用 One-vs-Rest 策略计算每个类别的 ROC

    保存: results/{model_name}_roc_curve.png
    """
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 将标签转为 one-hot 编码
    labels_onehot = np.zeros((len(labels), NUM_CLASSES))
    for i, label in enumerate(labels):
        labels_onehot[i, label] = 1

    all_auc = []

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{CLASS_NAMES_CN[i]} (AUC = {roc_auc:.4f})",
        )

    # 对角线（随机分类器基线）
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="随机基线")

    # 宏平均 AUC
    macro_auc = np.mean(all_auc)

    ax.set_xlabel("假阳性率 (FPR)", fontsize=12)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title(
        f"{model_name} — ROC 曲线 (宏平均 AUC = {macro_auc:.4f})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"{model_name}_roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📉 ROC 曲线已保存: {save_path}")

    return macro_auc


# ============================================================
# 图表 6：正确/错误预测样例
# ============================================================

def plot_prediction_examples(model, dataloader, model_name: str,
                             device=DEVICE, n_examples: int = 3):
    """
    展示正确预测和错误预测的样例

    保存:
        - results/{model_name}_correct_examples.png
        - results/{model_name}_error_examples.png
    """
    model.eval()

    correct_samples = {i: [] for i in range(NUM_CLASSES)}
    error_samples = []

    # 用于反归一化显示原图
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, preds = outputs.max(1)
            preds = preds.cpu()

            for i in range(len(labels)):
                img = images[i].cpu()
                # 反归一化
                img = img * std + mean
                img = img.clamp(0, 1)
                img_np = img.permute(1, 2, 0).numpy()

                true_label = labels[i].item()
                pred_label = preds[i].item()

                if true_label == pred_label:
                    if len(correct_samples[true_label]) < n_examples:
                        correct_samples[true_label].append((img_np, true_label, pred_label))
                else:
                    error_samples.append((img_np, true_label, pred_label))

    # --- 正确预测样例 ---
    fig, axes = plt.subplots(NUM_CLASSES, n_examples, figsize=(n_examples * 3, NUM_CLASSES * 3))
    for row in range(NUM_CLASSES):
        for col in range(n_examples):
            ax = axes[row][col] if NUM_CLASSES > 1 else axes[col]
            if col < len(correct_samples[row]):
                img_np, true_l, pred_l = correct_samples[row][col]
                ax.imshow(img_np)
                ax.set_title(f"✅ {CLASS_NAMES_CN[pred_l]}", fontsize=10, color="green")
            else:
                ax.text(0.5, 0.5, "无样本", ha="center", va="center")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_NAMES_CN[row], fontsize=12, rotation=0, labelpad=50)

    plt.suptitle(f"{model_name} — 正确预测样例", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"{model_name}_correct_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🖼️ 正确预测样例已保存: {save_path}")

    # --- 错误预测样例 ---
    n_errors = min(len(error_samples), 8)
    if n_errors > 0:
        cols = min(4, n_errors)
        rows = (n_errors + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(n_errors):
            r, c = idx // cols, idx % cols
            img_np, true_l, pred_l = error_samples[idx]
            axes[r][c].imshow(img_np)
            axes[r][c].set_title(
                f"真实: {CLASS_NAMES_CN[true_l]}\n预测: {CLASS_NAMES_CN[pred_l]}",
                fontsize=9, color="red",
            )
            axes[r][c].axis("off")

        # 隐藏多余的子图
        for idx in range(n_errors, rows * cols):
            r, c = idx // cols, idx % cols
            axes[r][c].axis("off")

        plt.suptitle(f"{model_name} — 错误预测样例", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, f"{model_name}_error_examples.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ❌ 错误预测样例已保存: {save_path}")
    else:
        print(f"  🎉 没有错误预测！（太完美了）")


# ============================================================
# 图表 7：两模型对比柱状图
# ============================================================

def plot_model_comparison(results: dict):
    """
    绘制 SimpleCNN vs ResNet18 的性能对比柱状图

    参数:
        results: {model_name: {'accuracy': float, 'macro_auc': float, ...}, ...}

    保存: results/model_comparison.png
    """
    model_names = list(results.keys())
    metrics = ["accuracy", "macro_f1", "macro_auc"]
    metric_labels = ["准确率", "宏平均 F1", "宏平均 AUC"]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width - width / 2, values, width,
                      label=model_name, color=colors[i % len(colors)])

        # 在柱子上显示数值
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=10,
            )

    ax.set_ylabel("分数", fontsize=12)
    ax.set_title("模型性能对比", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 模型对比图已保存: {save_path}")


# ============================================================
# 完整评估流程
# ============================================================

def get_tta_transforms():
    """
    获取测试时增强 (TTA) 的变换列表

    原理：对同一张图片应用多种变换，分别预测，取概率平均值
    就像考试时把答案检查 5 遍再交卷
    """
    return [
        # 变换 1：原图（和验证集相同）
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # 变换 2：水平翻转
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # 变换 3：轻微放大后中心裁剪
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # 变换 4：顺时针旋转 10°
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # 变换 5：逆时针旋转 10°
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]


def get_predictions_tta(model, dataset, device):
    """
    使用测试时增强 (TTA) 收集预测结果

    对每张图片应用 5 种变换，分别过模型，然后取概率平均值

    参数:
        model: 训练好的模型
        dataset: LogoDataset 实例（需要原始 samples 列表）
        device: 计算设备

    返回:
        all_labels, all_preds, all_probs
    """
    model.eval()
    tta_transforms = get_tta_transforms()
    n_tta = len(tta_transforms)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for img_path, label in dataset.samples:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

            # 对同一张图片应用多种变换，收集所有预测概率
            probs_sum = torch.zeros(NUM_CLASSES)
            for t in tta_transforms:
                img_tensor = t(image).unsqueeze(0).to(device)
                output = model(img_tensor)
                prob = F.softmax(output, dim=1)[0].cpu()
                probs_sum += prob

            # 取平均
            avg_probs = probs_sum / n_tta
            all_labels.append(label)
            all_probs.append(avg_probs.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = all_probs.argmax(axis=1)

    return all_labels, all_preds, all_probs


def evaluate_model(model, test_loader, model_name: str, use_tta: bool = True):
    """
    对一个模型执行完整评估，生成所有图表

    参数:
        model: 训练好的模型
        test_loader: 测试集 DataLoader
        model_name: 模型名称
        use_tta: 是否使用测试时增强

    返回:
        result_dict: {'accuracy': float, 'macro_f1': float, 'macro_auc': float}
    """
    print(f"\n{'='*60}")
    print(f"📊 评估模型: {model_name}")
    print(f"   TTA: {'开启 (5视角平均)' if use_tta else '关闭'}")
    print(f"{'='*60}")

    model = model.to(DEVICE)

    # 收集预测结果
    if use_tta and hasattr(test_loader.dataset, 'samples'):
        print("  🔄 正在进行测试时增强 (TTA)...")
        labels, preds, probs = get_predictions_tta(model, test_loader.dataset, DEVICE)
    else:
        labels, preds, probs = get_predictions(model, test_loader, DEVICE)

    # 计算总准确率
    accuracy = (preds == labels).sum() / len(labels)
    print(f"\n  🎯 测试集准确率: {accuracy:.4f} ({(preds == labels).sum()}/{len(labels)})")

    # 生成各种图表
    plot_confusion_matrix(labels, preds, model_name)
    report = print_classification_report(labels, preds, model_name)
    macro_auc = plot_roc_curves(labels, probs, model_name)
    plot_prediction_examples(model, test_loader, model_name)

    # 计算宏平均 F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    print(f"  📊 宏平均 F1: {f1:.4f}")

    return {"accuracy": accuracy, "macro_f1": f1, "macro_auc": macro_auc}
