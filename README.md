# 🧋 饮料品牌 LOGO 智能分类系统

基于卷积神经网络（CNN）的饮料品牌 LOGO 图像分类项目。使用 PyTorch 框架，对**喜茶、霸王茶姬、蜜雪冰城、瑞幸咖啡**四个品牌的 LOGO 进行自动识别分类。

## 📋 项目简介

本项目是人工智能课程的期末项目，主要工作包括：

1. **自构数据集**：通过网络爬虫和手动收集，构建了包含 4 个饮料品牌 LOGO 的图像数据集
2. **模型对比**：实现了两种 CNN 模型并进行对比实验
   - **SimpleCNN**：从零搭建的 3 层卷积神经网络（基线模型）
   - **ResNet50**：基于 ImageNet 预训练模型的迁移学习（主模型），解冻 layer3 + layer4
3. **完整评估**：使用准确率、F1 值、混淆矩阵、ROC/AUC 等多种指标评估模型性能
4. **GUI 演示**：提供图形界面，支持拖拽图片进行实时双模型对比预测

## 📁 项目结构

```
├── data/                          # 数据目录
│   ├── raw/                       # 原始图片（按品牌分类）
│   │   ├── heytea/                # 喜茶
│   │   ├── chagee/                # 霸王茶姬
│   │   ├── mixue/                 # 蜜雪冰城
│   │   └── luckin/                # 瑞幸咖啡
│   ├── train/                     # 训练集（自动划分）
│   ├── val/                       # 验证集（自动划分）
│   └── test/                      # 测试集（自动划分）
├── models/                        # 训练好的模型权重
├── results/                       # 实验结果（图表、报告）
├── scripts/
│   ├── scrape_images.py           # 图片爬虫脚本
│   ├── clean_data.py              # 图片自动预筛理（去重、去损坏）
│   └── reorder_data.py            # 图片格式统一与重编号
├── src/
│   ├── dataset.py                 # 数据集加载与预处理
│   ├── model.py                   # 模型定义（SimpleCNN + ResNet50）
│   ├── train.py                   # 训练脚本
│   ├── evaluate.py                # 评估与可视化
│   ├── predict.py                 # 单张图片预测
│   └── gui.py                     # GUI 图形界面（拖拽识别）
├── config.py                      # 全局配置
├── main.py                        # 主入口（一键训练+评估）
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

## 🛠️ 环境配置

### 系统要求

- Python 3.10+
- NVIDIA GPU（推荐，支持 CUDA 12.8+）
- Windows / Linux / macOS

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/YOUR_USERNAME/milk-tea-logo-classifier.git
cd milk-tea-logo-classifier
```

2. **安装 PyTorch（GPU 版）**

如果你的 GPU 是 RTX 40 系列或更早：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

如果你的 GPU 是 RTX 50 系列（Blackwell 架构）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. **安装其他依赖**
```bash
pip install -r requirements.txt
```

## 📊 数据准备

### 方式一：使用爬虫自动下载

```bash
# 爬取所有品牌
python scripts/scrape_images.py

# 只爬取指定品牌
python scripts/scrape_images.py --brand heytea

# 只使用百度源
python scripts/scrape_images.py --source baidu
```

### 方式二：手动收集

将图片按品牌放入对应文件夹：
```
data/raw/heytea/    ← 喜茶 LOGO 图片
data/raw/chagee/    ← 霸王茶姬 LOGO 图片
data/raw/mixue/     ← 蜜雪冰城 LOGO 图片
data/raw/luckin/    ← 瑞幸咖啡 LOGO 图片
```

> ⚠️ 建议每个品牌至少 60 张图片，支持 jpg/jpeg/png/webp/bmp 格式。

### 数据预处理（可选）

```bash
# 自动清理：去除损坏、过小、重复的图片
python scripts/clean_data.py

# 统一格式为 JPG 并重新编号
python scripts/reorder_data.py
```

## 🚀 运行

### 一键训练 + 评估

```bash
python main.py
```

该命令会自动执行：
1. 数据集划分（70% 训练 / 15% 验证 / 15% 测试）
2. 训练 SimpleCNN 基线模型
3. 训练 ResNet50 迁移学习模型
4. 在测试集上评估两个模型
5. 生成所有实验图表到 `results/` 目录

### 单张图片预测

```bash
python src/predict.py path/to/your/logo.jpg
python src/predict.py path/to/your/logo.jpg --model simple_cnn
```

### GUI 图形界面

```bash
python src/gui.py
```

支持拖拽图片到窗口进行实时识别，同时展示 SimpleCNN 和 ResNet50 的预测对比。

## 📈 实验结果

训练完成后，`results/` 目录下会生成以下文件：

| 文件 | 描述 |
|------|------|
| `data_distribution.png` | 各品牌数据集分布柱状图 |
| `sample_images.png` | 各品牌示例图片 |
| `SimpleCNN_loss_curve.png` | SimpleCNN 损失曲线 |
| `SimpleCNN_acc_curve.png` | SimpleCNN 准确率曲线 |
| `SimpleCNN_confusion_matrix.png` | SimpleCNN 混淆矩阵 |
| `SimpleCNN_roc_curve.png` | SimpleCNN ROC 曲线 |
| `ResNet50_loss_curve.png` | ResNet50 损失曲线 |
| `ResNet50_acc_curve.png` | ResNet50 准确率曲线 |
| `ResNet50_confusion_matrix.png` | ResNet50 混淆矩阵 |
| `ResNet50_roc_curve.png` | ResNet50 ROC 曲线 |
| `model_comparison.png` | 两模型性能对比 |

## ⚙️ 配置修改

所有超参数在 `config.py` 中统一管理，可按需调整：

```python
BATCH_SIZE = 32            # 批次大小
LEARNING_RATE = 0.001      # 学习率
NUM_EPOCHS = 50            # 训练轮数
IMAGE_SIZE = 224           # 输入图片尺寸
```

## 📝 License

本项目仅用于课程学习目的。
