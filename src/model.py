"""
模型定义模块
包含三个模型：
1. SimpleCNN — 从零搭建的简单卷积神经网络（基线模型）
2. ResNet18 — 基于预训练模型的轻量迁移学习
3. ResNet50 — 基于预训练模型的重量级迁移学习（主模型）
"""

import torch
import torch.nn as nn
from torchvision import models

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, IMAGE_SIZE


# ============================================================
# 模型一：SimpleCNN（自建基线模型）
# ============================================================

class SimpleCNN(nn.Module):
    """简单的三层卷积神经网络"""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ============================================================
# 模型二：ResNet18 迁移学习
# ============================================================

def create_resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                    freeze_backbone: bool = True):
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        print("  ✅ 已加载 ImageNet 预训练权重 (ResNet18)")
    else:
        model = models.resnet18(weights=None)
        print("  ⚠️ 未使用预训练权重，从零训练 (ResNet18)")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        print("  🔒 骨干网络已冻结（layer4 除外）")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    print(f"  🔄 全连接层已替换: {in_features} → {num_classes}")
    return model

# ============================================================
# 模型三：ResNet50 迁移学习 (进阶升级版)
# ============================================================

def create_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                    freeze_backbone: bool = True):
    """
    创建基于 ResNet50 的进阶版迁移学习模型
    为了榨干 5070Ti 的性能并提高准确率，我们选择解冻 layer3 和 layer4 两个 block。
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        print("  ✅ 已加载 ImageNet 预训练权重 (ResNet50, V2)")
    else:
        model = models.resnet50(weights=None)
        print("  ⚠️ 未使用预训练权重，从零训练 (ResNet50)")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # 为了更强的拟合能力，解冻 layer3 和 layer4
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        print("  🔓 骨干网络部分解冻（layer3, layer4 参与训练）")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    print(f"  🔄 全连接层已替换: {in_features} → {num_classes}")
    return model


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }

def get_model(model_name: str, **kwargs) -> nn.Module:
    if model_name == "simple_cnn":
        model = SimpleCNN(**kwargs)
    elif model_name == "resnet18":
        model = create_resnet18(**kwargs)
    elif model_name == "resnet50":
        model = create_resnet50(**kwargs)
    else:
        raise ValueError(f"未知模型: {model_name}，可选: 'simple_cnn', 'resnet18', 'resnet50'")

    params = count_parameters(model)
    print(f"  📐 参数量: 总计 {params['total']:,} | "
          f"可训练 {params['trainable']:,} | 冻结 {params['frozen']:,}")
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 测试模型定义")
    print("=" * 60)
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    resnet = get_model("resnet50")
    output = resnet(dummy_input)
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
