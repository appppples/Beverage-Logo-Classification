"""
数据集模块
负责数据集划分、数据增强、加载和可视化
"""

import os
import sys
import shutil
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, RESULT_DIR,
    CLASSES, CLASS_NAMES_CN, IMAGE_SIZE, BATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ============================================================
# 数据增强 / 变换定义
# ============================================================

def get_train_transforms():
    """训练集的数据增强变换"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 先放大一点
        transforms.RandomCrop(IMAGE_SIZE),                      # 随机裁剪到目标尺寸
        transforms.RandomHorizontalFlip(p=0.5),                 # 50% 概率水平翻转
        transforms.RandomRotation(degrees=15),                  # 随机旋转 ±15°
        transforms.ColorJitter(                                 # 颜色抖动
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(                                # 随机仿射变换
            degrees=0, translate=(0.1, 0.1)
        ),
        transforms.ToTensor(),                                  # 转为张量 [0,1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet 归一化
    ])


def get_val_transforms():
    """验证集/测试集的变换（不做增强）"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),             # 直接缩放
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# 数据集划分
# ============================================================

def split_dataset():
    """
    将 data/raw/ 中的图片按比例划分到 train/val/test 目录

    划分逻辑：
    1. 遍历 raw/ 下每个品牌文件夹
    2. 随机打乱图片列表
    3. 按 TRAIN_RATIO : VAL_RATIO : TEST_RATIO 切分
    4. 复制（不是移动）到对应目录

    如果 train/val/test 目录已有内容，会先清空再重新划分。
    """
    random.seed(RANDOM_SEED)

    # 清空旧的划分
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    split_stats = {}  # 记录每个品牌的划分数量

    for cls in CLASSES:
        raw_cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if not os.path.exists(raw_cls_dir):
            print(f"⚠️  {cls} 的原始数据目录不存在，跳过")
            continue

        # 获取所有有效图片文件
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [
            f for f in os.listdir(raw_cls_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        if len(images) == 0:
            print(f"⚠️  {cls} 没有图片文件，跳过")
            continue

        # 随机打乱
        random.shuffle(images)

        # 计算划分点
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # 剩余全给测试集
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # 复制到对应目录
        for split_name, split_dir, img_list in [
            ("train", TRAIN_DIR, train_imgs),
            ("val", VAL_DIR, val_imgs),
            ("test", TEST_DIR, test_imgs),
        ]:
            cls_split_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_split_dir, exist_ok=True)
            for img_name in img_list:
                src = os.path.join(raw_cls_dir, img_name)
                dst = os.path.join(cls_split_dir, img_name)
                shutil.copy2(src, dst)

        split_stats[cls] = {
            "total": n,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

        cn_name = CLASS_NAMES_CN[CLASSES.index(cls)]
        print(
            f"  {cn_name:　<5} ({cls:>8}): "
            f"总计 {n:>3} → "
            f"训练 {len(train_imgs):>3} | "
            f"验证 {len(val_imgs):>3} | "
            f"测试 {len(test_imgs):>3}"
        )

    return split_stats


# ============================================================
# PyTorch Dataset 类
# ============================================================

class LogoDataset(Dataset):
    """
    LOGO 图片数据集

    目录结构要求：
        root_dir/
            heytea/
                001.jpg
                002.jpg
            chagee/
                ...
    """

    def __init__(self, root_dir: str, transform=None):
        """
        参数:
            root_dir: 数据集根目录（如 data/train）
            transform: 图片变换/增强
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # [(图片路径, 标签索引), ...]
        self.classes = CLASSES

        for label_idx, cls in enumerate(CLASSES):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # 如果图片损坏，返回一张黑色图片
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# DataLoader 创建
# ============================================================

def create_dataloaders():
    """
    创建训练/验证/测试的 DataLoader

    返回:
        train_loader, val_loader, test_loader
    """
    train_dataset = LogoDataset(TRAIN_DIR, transform=get_train_transforms())
    val_dataset = LogoDataset(VAL_DIR, transform=get_val_transforms())
    test_dataset = LogoDataset(TEST_DIR, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"\n📊 数据集统计:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")

    return train_loader, val_loader, test_loader


# ============================================================
# 数据可视化
# ============================================================

def visualize_dataset_distribution(split_stats: dict):
    """
    绘制各品牌在训练/验证/测试集中的样本数量分布柱状图
    保存到 results/data_distribution.png
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    brands = []
    train_counts = []
    val_counts = []
    test_counts = []

    for cls in CLASSES:
        if cls in split_stats:
            cn_name = CLASS_NAMES_CN[CLASSES.index(cls)]
            brands.append(cn_name)
            train_counts.append(split_stats[cls]["train"])
            val_counts.append(split_stats[cls]["val"])
            test_counts.append(split_stats[cls]["test"])

    x = np.arange(len(brands))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, train_counts, width, label="训练集", color="#4ECDC4")
    bars2 = ax.bar(x, val_counts, width, label="验证集", color="#FFE66D")
    bars3 = ax.bar(x + width, test_counts, width, label="测试集", color="#FF6B6B")

    ax.set_xlabel("品牌", fontsize=12)
    ax.set_ylabel("图片数量", fontsize=12)
    ax.set_title("各品牌数据集分布", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(brands, fontsize=11)
    ax.legend(fontsize=11)

    # 在柱子上方显示数量
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "data_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 数据分布图已保存: {save_path}")


def visualize_sample_images():
    """
    展示每个品牌的示例图片（3×4 网格）
    保存到 results/sample_images.png
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    n_samples = 3  # 每个品牌展示 3 张

    fig, axes = plt.subplots(
        len(CLASSES), n_samples,
        figsize=(n_samples * 3, len(CLASSES) * 3)
    )

    for row, cls in enumerate(CLASSES):
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            continue

        images = [f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        random.seed(RANDOM_SEED)
        selected = random.sample(images, min(n_samples, len(images)))

        cn_name = CLASS_NAMES_CN[row]

        for col in range(n_samples):
            ax = axes[row][col] if len(CLASSES) > 1 else axes[col]
            if col < len(selected):
                img_path = os.path.join(cls_dir, selected[col])
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "加载失败", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "无图片", ha="center", va="center")

            ax.axis("off")
            if col == 0:
                ax.set_title(cn_name, fontsize=13, fontweight="bold")

    plt.suptitle("各品牌 LOGO 示例图片", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "sample_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🖼️ 示例图片已保存: {save_path}")


# ============================================================
# 模块测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📁 数据集划分")
    print("=" * 60)
    stats = split_dataset()

    if stats:
        print("\n" + "=" * 60)
        print("📊 生成数据可视化")
        print("=" * 60)
        visualize_dataset_distribution(stats)
        visualize_sample_images()
    else:
        print("\n⚠️  没有找到数据！请先运行爬虫或手动放入图片到 data/raw/ 目录")
