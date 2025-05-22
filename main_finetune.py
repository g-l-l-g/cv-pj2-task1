# caltech101_classification/main_finetune.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 从 caltech101_dataprep 包导入
from caltech101_dataprep import data_utils as dp_utils
from caltech101_dataprep import config as dp_config  # 数据准备的配置

# 本地模块
import config_train as train_cfg  # 训练配置
from engine import train_one_epoch, evaluate
from model.models import get_model
from utils.training_utils import save_checkpoint, load_checkpoint, setup_tensorboard_writer, get_optimizer


def main_finetune():
    print("Starting Finetuning Experiment...")
    print(f"Using device: {train_cfg.DEVICE}")

    # --- 1. 加载数据 ---
    # 更新数据准备配置（如果需要，比如BATCH_SIZE）
    dp_config.BATCH_SIZE = train_cfg.BATCH_SIZE  # 确保数据加载器使用训练配置中的batch_size
    train_loader, val_loader, test_loader, class_to_idx, num_data_classes, _ = dp_utils.get_dataloaders()

    if num_data_classes != train_cfg.NUM_CLASSES:
        raise ValueError(
            f"Mismatch in number of classes: data loader reports {num_data_classes}, "
            f"but train_cfg.NUM_CLASSES is {train_cfg.NUM_CLASSES}."
        )
    print(f"Data loaded: {num_data_classes} classes.")

    # --- 2. 构建模型 ---
    model = get_model(model_name=train_cfg.MODEL_NAME,
                      num_classes=train_cfg.NUM_CLASSES,
                      pretrained=True)
    model = model.to(train_cfg.DEVICE)

    # --- 3. 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model,
        config_optimizer=train_cfg.FINETUNE_OPTIMIZER,
        learning_rate=train_cfg.FINETUNE_LEARNING_RATE_FC,
        weight_decay=train_cfg.FINETUNE_WEIGHT_DECAY,
        finetune_fc_lr=train_cfg.FINETUNE_LEARNING_RATE_FC,
        finetune_base_lr=train_cfg.FINETUNE_LEARNING_RATE_BASE
    )

    scheduler = None
    if train_cfg.USE_LR_SCHEDULER:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=train_cfg.LR_SCHEDULER_FACTOR,
                                      patience=train_cfg.LR_SCHEDULER_PATIENCE, verbose=True)

    # --- 4. TensorBoard Writer ---
    tb_experiment_name_suffix = getattr(train_cfg, 'LOG_DIR_FINETUNE_EXPERIMENT_NAME',
                                        f"finetune_{train_cfg.MODEL_NAME}")
    writer = setup_tensorboard_writer(train_cfg.LOG_DIR_FINETUNE, tb_experiment_name_suffix)

    # --- 5. 训练循环 ---
    best_val_accuracy = 0.0
    start_epoch = 0
    test_acc = 0.0

    print(f"Starting training from epoch {start_epoch + 1} for {train_cfg.EPOCHS} epochs.")
    for epoch in range(start_epoch, train_cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, train_cfg.DEVICE, epoch, writer
        )
        val_loss, val_acc = evaluate(
            model, criterion, val_loader, train_cfg.DEVICE, epoch, writer, phase="Val"
        )

        if scheduler:
            scheduler.step(val_acc)  # ReduceLROnPlateau 需要一个指标来判断

        # 保存最佳模型
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.4f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'class_to_idx': class_to_idx,
            'model_name': train_cfg.MODEL_NAME
        }, is_best, train_cfg.CHECKPOINT_DIR_FINETUNE, best_filename=train_cfg.BEST_MODEL_NAME_FINETUNE)

    print(f"Finetuning finished. Best Validation Accuracy: {best_val_accuracy:.4f}")

    # --- 6. 在测试集上评估最佳模型 ---
    print("\nEvaluating on Test Set using the best model...")
    best_model_path = os.path.join(train_cfg.CHECKPOINT_DIR_FINETUNE, train_cfg.BEST_MODEL_NAME_FINETUNE)
    if os.path.exists(best_model_path):
        # 创建一个新模型实例以加载状态（或直接使用当前模型，但加载状态更干净）
        # pretrained=False, 我们将加载我们训练的权重
        test_model = get_model(model_name=train_cfg.MODEL_NAME, num_classes=train_cfg.NUM_CLASSES, pretrained=False)
        load_checkpoint(best_model_path, test_model)  # 只加载模型权重，不需要优化器状态
        test_model = test_model.to(train_cfg.DEVICE)
        test_loss, test_acc = evaluate(
            test_model, criterion, test_loader, train_cfg.DEVICE, train_cfg.EPOCHS, writer=None, phase="Test"
        )  # epoch=EPOCHS只是为了打印格式
        print(f"Test Set Results - Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        if writer:
            writer.add_hparams(
                {"lr_base": train_cfg.FINETUNE_LEARNING_RATE_BASE,
                 "lr_fc": train_cfg.FINETUNE_LEARNING_RATE_FC,
                 "optimizer": train_cfg.FINETUNE_OPTIMIZER,
                 "batch_size": train_cfg.BATCH_SIZE,
                 "model": train_cfg.MODEL_NAME},
                {"hparam/test_accuracy": test_acc,
                 "hparam/test_loss": test_loss,
                 "hparam/best_val_accuracy": best_val_accuracy}
            )
    else:
        print(f"Best model checkpoint not found at {best_model_path}")

    if writer:
        writer.close()

    return best_val_accuracy, test_acc


if __name__ == '__main__':
    main_finetune()
