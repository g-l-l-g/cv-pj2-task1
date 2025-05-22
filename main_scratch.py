# caltech101_classification/main_scratch.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 从 caltech101_dataprep 包导入
from caltech101_dataprep import data_utils as dp_utils
from caltech101_dataprep import config as dp_config

# 本地模块
import config_train as train_cfg
from model.models import get_model
from utils.training_utils import save_checkpoint, load_checkpoint, setup_tensorboard_writer, get_optimizer
from engine import train_one_epoch, evaluate


def main_scratch():
    print("Starting Training from Scratch Experiment...")
    print(f"Using device: {train_cfg.DEVICE}")

    # --- 1. 加载数据 ---
    dp_config.BATCH_SIZE = train_cfg.BATCH_SIZE
    train_loader, val_loader, test_loader, class_to_idx, num_data_classes, _ = \
        dp_utils.get_dataloaders()

    if num_data_classes != train_cfg.NUM_CLASSES:
        raise ValueError(
            f"Mismatch in number of classes: data loader reports {num_data_classes}, "
            f"but train_cfg.NUM_CLASSES is {train_cfg.NUM_CLASSES}."
        )
    print(f"Data loaded: {num_data_classes} classes.")

    # --- 2. 构建模型 ---
    model = get_model(model_name=train_cfg.MODEL_NAME,
                      num_classes=train_cfg.NUM_CLASSES,
                      pretrained=False)  # pretrained=False for training from scratch
    model = model.to(train_cfg.DEVICE)

    # --- 3. 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model,
        config_optimizer=train_cfg.SCRATCH_OPTIMIZER,
        learning_rate=train_cfg.SCRATCH_LEARNING_RATE,
        weight_decay=train_cfg.SCRATCH_WEIGHT_DECAY
    )  # 不传入 finetune_fc_lr 和 finetune_base_lr

    scheduler = None
    if train_cfg.USE_LR_SCHEDULER:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=train_cfg.LR_SCHEDULER_FACTOR,
                                      patience=train_cfg.LR_SCHEDULER_PATIENCE, verbose=True)

    # --- 4. TensorBoard Writer ---
    writer = setup_tensorboard_writer(train_cfg.LOG_DIR_SCRATCH, f"scratch_{train_cfg.MODEL_NAME}")

    # --- 5. 训练循环 ---
    best_val_accuracy = 0.0
    start_epoch = 0

    print(f"Starting training from epoch {start_epoch + 1} for {train_cfg.EPOCHS} epochs.")
    for epoch in range(start_epoch, train_cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, train_cfg.DEVICE, epoch, writer
        )
        val_loss, val_acc = evaluate(
            model, criterion, val_loader, train_cfg.DEVICE, epoch, writer, phase="Val"
        )

        if scheduler:
            scheduler.step(val_acc)

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
        }, is_best, train_cfg.CHECKPOINT_DIR_SCRATCH, best_filename=train_cfg.BEST_MODEL_NAME_SCRATCH)

    print(f"Training from scratch finished. Best Validation Accuracy: {best_val_accuracy:.4f}")

    # --- 6. 在测试集上评估最佳模型 ---
    print("\nEvaluating on Test Set using the best model...")
    best_model_path = os.path.join(train_cfg.CHECKPOINT_DIR_SCRATCH, train_cfg.BEST_MODEL_NAME_SCRATCH)
    if os.path.exists(best_model_path):
        test_model = get_model(model_name=train_cfg.MODEL_NAME, num_classes=train_cfg.NUM_CLASSES, pretrained=False)
        load_checkpoint(best_model_path, test_model)
        test_model = test_model.to(train_cfg.DEVICE)
        test_loss, test_acc = evaluate(
            test_model, criterion, test_loader, train_cfg.DEVICE, train_cfg.EPOCHS, writer=None, phase="Test"
        )
        print(f"Test Set Results - Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        if writer:
            writer.add_hparams(
                {"lr": train_cfg.SCRATCH_LEARNING_RATE,
                 "optimizer": train_cfg.SCRATCH_OPTIMIZER,
                 "batch_size": train_cfg.BATCH_SIZE,
                 "model": train_cfg.MODEL_NAME,
                 "pretrained": False},
                {"hparam/test_accuracy": test_acc,
                 "hparam/test_loss": test_loss,
                 "hparam/best_val_accuracy": best_val_accuracy},
            )
    else:
        print(f"Best model checkpoint not found at {best_model_path}")

    if writer:
        writer.close()


if __name__ == '__main__':
    main_scratch()

