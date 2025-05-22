# caltech101_classification/config_train.py
import torch

# --- 基本配置 ---
MODEL_NAME = "resnet18"  # 可选 'resnet18', 'alexnet', etc.
NUM_CLASSES = 101        # Caltech-101 有101个类别 (不含背景)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 训练超参数 ---
EPOCHS = 15
BATCH_SIZE = 32  # 应与 caltech101_dataprep/config.py 中的 BATCH_SIZE 一致或在此覆盖

# --- 微调 (Finetuning) 特定配置 ---
FINETUNE_LEARNING_RATE_BASE = 1e-4  # 预训练层学习率
FINETUNE_LEARNING_RATE_FC = 1e-3   # 新分类层学习率
FINETUNE_OPTIMIZER = "Adam"        # 可选 "SGD", "Adam", "AdamW"
FINETUNE_WEIGHT_DECAY = 1e-4

# --- 从零训练 (Scratch) 特定配置 ---
SCRATCH_LEARNING_RATE = 1e-3
SCRATCH_OPTIMIZER = "Adam"         # 可选 "SGD", "Adam", "AdamW"
SCRATCH_WEIGHT_DECAY = 1e-4

# --- 学习率调度器 (可选) ---
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5  # ReduceLROnPlateau: 在N个epoch验证集性能没提升后降低LR
LR_SCHEDULER_FACTOR = 0.1  # ReduceLROnPlateau: LR降低因子

# --- 路径配置 ---
CHECKPOINT_DIR_FINETUNE = f"./checkpoints/finetune_{MODEL_NAME}"
CHECKPOINT_DIR_SCRATCH = f"./checkpoints/scratch_{MODEL_NAME}"
LOG_DIR_FINETUNE = f"./runs/finetune_{MODEL_NAME}"
LOG_DIR_SCRATCH = f"./runs/scratch_{MODEL_NAME}"
BEST_MODEL_NAME_FINETUNE = f"best_model_finetune_{MODEL_NAME}.pth.tar"
BEST_MODEL_NAME_SCRATCH = f"best_model_scratch_{MODEL_NAME}.pth.tar"

