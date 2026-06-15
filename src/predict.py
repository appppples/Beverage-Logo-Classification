"""
单张图片预测模块
加载训练好的模型，对任意一张图片进行品牌分类预测
适合用于论文演示截图
"""

import os
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, MODEL_DIR, IMAGE_SIZE, NUM_CLASSES,
    CLASSES, CLASS_NAMES_CN, IMAGENET_MEAN, IMAGENET_STD, RESULT_DIR,
)
from src.model import get_model

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 预测函数
# ============================================================

def load_trained_model(model_name: str = "resnet18"):
    """加载训练好的模型权重"""
    model_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型（不加载预训练权重，因为我们要加载自己训练的权重）
    if model_name == "resnet18":
        model = get_model("resnet18", pretrained=False, freeze_backbone=False)
    elif model_name == "resnet50":
        model = get_model("resnet50", pretrained=False, freeze_backbone=False)
    else:
        model = get_model("simple_cnn")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    best_acc = checkpoint.get("best_val_acc", "未知")
    print(f"  ✅ 模型已加载: {model_name} (验证准确率: {best_acc})")
    return model


def predict_single_image(model, image_path: str):
    """
    对单张图片进行预测

    返回:
        pred_class: 预测的品牌名（中文）
        pred_idx: 预测的类别索引
        probs: 各类别的概率 (list of float)
    """
    # 图片变换（与验证集相同）
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # 加 batch 维度

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = CLASS_NAMES_CN[pred_idx]
    probs_list = probs.cpu().tolist()

    return pred_class, pred_idx, probs_list


def predict_and_visualize(model, image_path: str, save: bool = True):
    """
    预测并可视化结果（原图 + 概率柱状图）

    保存: results/prediction_{filename}.png
    """
    pred_class, pred_idx, probs = predict_single_image(model, image_path)

    # 创建并排图：左边原图，右边概率柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # 左边：原图
    img = Image.open(image_path).convert("RGB")
    ax1.imshow(img)
    ax1.set_title(f"预测结果: {pred_class}", fontsize=14, fontweight="bold",
                  color="green" if max(probs) > 0.8 else "orange")
    ax1.axis("off")

    # 右边：各品牌概率柱状图
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3"]
    bars = ax2.barh(CLASS_NAMES_CN, probs, color=colors[:NUM_CLASSES])
    ax2.set_xlabel("预测概率", fontsize=12)
    ax2.set_title("各品牌概率分布", fontsize=13, fontweight="bold")
    ax2.set_xlim([0, 1.1])

    # 在柱子右侧显示概率值
    for bar, prob in zip(bars, probs):
        ax2.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{prob:.2%}", va="center", fontsize=11,
        )

    plt.tight_layout()

    if save:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(RESULT_DIR, f"prediction_{filename}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📸 预测结果已保存: {save_path}")

    plt.close()
    return pred_class, probs


# ============================================================
# 命令行入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="单张 LOGO 图片预测")
    parser.add_argument("image", type=str, help="图片路径")
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=["simple_cnn", "resnet18", "resnet50"],
        help="使用的模型（默认 resnet50）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ 图片不存在: {args.image}")
        return

    print("=" * 60)
    print("🔍 LOGO 品牌预测")
    print("=" * 60)

    model = load_trained_model(args.model)
    pred_class, probs = predict_and_visualize(model, args.image)

    print(f"\n  🎯 预测结果: {pred_class}")
    print(f"  📊 各品牌概率:")
    for name, prob in zip(CLASS_NAMES_CN, probs):
        bar = "█" * int(prob * 30)
        print(f"     {name:　<5} {prob:>6.2%} {bar}")


if __name__ == "__main__":
    main()
