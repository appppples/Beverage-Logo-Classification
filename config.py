"""
全局配置文件
定义项目中所有超参数、路径和品牌映射
"""

import os
import sys
import torch

# 修复 Windows 终端 GBK 编码问题（支持 emoji 和中文输出）
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 路径配置
# ============================================================
# 项目根目录（自动定位到 config.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 输出路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# 自动创建所有必要目录
for d in [RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 品牌配置
# ============================================================
# 品牌英文文件夹名 → 中文显示名
BRAND_NAMES = {
    "heytea": "喜茶",
    "chagee": "霸王茶姬",
    "mixue": "蜜雪冰城",
    "luckin": "瑞幸咖啡",
}

# 类别列表（与标签索引对应）
CLASSES = list(BRAND_NAMES.keys())       # ['heytea', 'chagee', 'mixue', 'luckin']
CLASS_NAMES_CN = list(BRAND_NAMES.values())  # ['喜茶', '霸王茶姬', '蜜雪冰城', '瑞幸咖啡']
NUM_CLASSES = len(CLASSES)                # 4

# 为每个品牌的原始数据创建子目录
for cls in CLASSES:
    os.makedirs(os.path.join(RAW_DATA_DIR, cls), exist_ok=True)

# ============================================================
# 爬虫关键词配置
# ============================================================
SCRAPE_KEYWORDS = {
    "heytea": ["喜茶LOGO", "喜茶标志", "HEYTEA logo", "喜茶门店招牌", "喜茶杯子"],
    "chagee": ["霸王茶姬LOGO", "霸王茶姬标志", "CHAGEE logo", "霸王茶姬门店", "霸王茶姬杯子"],
    "mixue": ["蜜雪冰城LOGO", "蜜雪冰城雪王", "MIXUE logo", "蜜雪冰城门店招牌", "蜜雪冰城标志"],
    "luckin": ["瑞幸咖啡LOGO", "瑞幸咖啡标志", "Luckin Coffee logo", "瑞幸咖啡门店", "瑞幸咖啡杯子"],
}

# 每个关键词最大下载数量
SCRAPE_MAX_PER_KEYWORD = 30

# ============================================================
# 数据处理参数
# ============================================================
IMAGE_SIZE = 224           # 输入图片尺寸（ResNet 标准）
TRAIN_RATIO = 0.70         # 训练集比例
VAL_RATIO = 0.15           # 验证集比例
TEST_RATIO = 0.15          # 测试集比例
RANDOM_SEED = 42           # 随机种子（保证可复现）

# ============================================================
# 训练超参数
# ============================================================
BATCH_SIZE = 32            # 批次大小
LEARNING_RATE = 0.001      # 初始学习率
NUM_EPOCHS = 50            # 最大训练轮数
WEIGHT_DECAY = 1e-4        # L2 正则化系数
LR_STEP_SIZE = 10          # 学习率衰减步长（每 N 轮衰减一次）
LR_GAMMA = 0.1             # 学习率衰减比例
EARLY_STOP_PATIENCE = 10   # 早停耐心值（验证集 N 轮不提升则停止）
LABEL_SMOOTHING = 0.1      # 标签平滑系数（防止模型过度自信）
MIXUP_ALPHA = 0.2          # Mixup 增强的 Beta 分布参数（0 表示关闭）

# ============================================================
# 设备配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ImageNet 归一化参数（用于预训练模型）
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
