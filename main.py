"""
主入口脚本
一键完成全部流程：数据划分 → 训练两个模型 → 评估对比 → 生成所有图表
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, RAW_DATA_DIR, CLASSES, BRAND_NAMES

from src.dataset import split_dataset, create_dataloaders, visualize_dataset_distribution, visualize_sample_images
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_curves, plot_model_comparison


def check_data():
    """检查数据是否已准备好"""
    total_images = 0
    print("\n📁 数据检查:")
    for cls in CLASSES:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])
        else:
            count = 0
        cn_name = BRAND_NAMES[cls]
        status = "✅" if count >= 30 else ("⚠️" if count > 0 else "❌")
        print(f"  {status} {cn_name:　<5} ({cls}): {count} 张")
        total_images += count

    if total_images == 0:
        print("\n❌ 没有找到任何图片！")
        print("   请先执行以下操作之一：")
        print("   1. 运行爬虫: python scripts/scrape_images.py")
        print("   2. 手动将图片放到 data/raw/ 对应品牌文件夹中")
        return False

    if total_images < 40:
        print(f"\n⚠️  图片总数只有 {total_images} 张，建议至少每个品牌 30 张以上")
        response = input("   是否继续？(y/n): ").strip().lower()
        if response != 'y':
            return False

    return True


def main():
    start_time = time.time()

    print("=" * 60)
    print("🧋 奶茶品牌 LOGO 分类 — 完整训练流程")
    print(f"   设备: {DEVICE}")
    print("=" * 60)

    # ========== 第一步：检查数据 ==========
    if not check_data():
        return

    # ========== 第一步：划分数据集 ==========
    print("\n" + "=" * 60)
    print("📁 第一步：划分数据集")
    print("=" * 60)
    split_stats = split_dataset()

    if not split_stats:
        print("❌ 数据划分失败！")
        return

    # ========== 第二步：可视化数据 ==========
    print("\n" + "=" * 60)
    print("📊 第二步：数据可视化")
    print("=" * 60)
    visualize_dataset_distribution(split_stats)
    visualize_sample_images()

    # ========== 第三步：创建 DataLoader ==========
    print("\n" + "=" * 60)
    print("📦 第三步：创建 DataLoader")
    print("=" * 60)
    train_loader, val_loader, test_loader = create_dataloaders()

    # ========== 第四步：训练 SimpleCNN ==========
    print("\n" + "=" * 60)
    print("🏗️ 第四步：训练 SimpleCNN（基线模型）")
    print("=" * 60)
    simple_cnn = get_model("simple_cnn")
    history_cnn = train_model(simple_cnn, train_loader, val_loader, model_name="simple_cnn")
    plot_training_curves(history_cnn, "SimpleCNN")

    # ========== 第五步：训练 ResNet50 ==========
    print("\n" + "=" * 60)
    print("🚀 第五步：训练 ResNet50（迁移学习）")
    print("=" * 60)
    resnet = get_model("resnet50")
    history_resnet = train_model(resnet, train_loader, val_loader, model_name="resnet50")
    plot_training_curves(history_resnet, "ResNet50")

    # ========== 第六步：评估两个模型 ==========
    print("\n" + "=" * 60)
    print("📊 第六步：模型评估")
    print("=" * 60)
    result_cnn = evaluate_model(simple_cnn, test_loader, "SimpleCNN")
    result_resnet = evaluate_model(resnet, test_loader, "ResNet50")

    # ========== 第七步：模型对比 ==========
    print("\n" + "=" * 60)
    print("📊 第七步：模型对比")
    print("=" * 60)
    comparison_results = {
        "SimpleCNN": result_cnn,
        "ResNet50": result_resnet,
    }
    plot_model_comparison(comparison_results)

    # ========== 完成 ==========
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 全部完成！")
    print(f"   总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\n   📊 SimpleCNN 测试准确率: {result_cnn['accuracy']:.4f}")
    print(f"   🚀 ResNet50 测试准确率: {result_resnet['accuracy']:.4f}")
    print(f"\n   所有结果已保存到 results/ 目录")
    print(f"   模型权重已保存到 models/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
