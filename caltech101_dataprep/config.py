# caltech101_dataprep/config.py

# --- 数据集路径 ---
# 修改为'101_ObjectCategories' 文件夹的实际路径
DATA_ROOT_DIR = r'D:\python object\computer vision\project2\task1\caltech-101\101_ObjectCategories\101_ObjectCategories'

# --- 划分参数 ---
NUM_TRAIN_PER_CLASS = 25  # 每类用于训练的样本数
NUM_VAL_PER_CLASS = 5     # 每类用于验证的样本数

# --- 通用参数 ---
RANDOM_SEED = 42          # 保证可复现性
IMAGE_SIZE = (224, 224)   # 预训练模型输入尺寸
BATCH_SIZE = 32
NUM_WORKERS = 4           # DataLoader 的工作进程数
CLASS_MAPPING_FILE = 'caltech101_class_to_idx.json'  # 类别映射文件名

# --- ImageNet 统计数据 (用于标准化) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 要排除的文件夹 ---
EXCLUDE_DIRS = ["BACKGROUND_Google"]
