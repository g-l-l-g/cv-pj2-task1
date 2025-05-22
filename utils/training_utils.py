# caltech101_classification/utils/training_utils.py
import os
import shutil
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar", best_filename="model_best.pth.tar"):
    """保存模型检查点，如果is_best为True，则额外保存为最佳模型。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        print(f"Saved new best model to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载模型检查点。"""
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)  # CPU or GPU
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)  # 使用 .get 提供默认值
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_val_acc {best_val_accuracy:.4f})")
        return start_epoch, best_val_accuracy
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return 0, 0.0


def setup_tensorboard_writer(log_dir_base, experiment_name):
    """设置TensorBoard SummaryWriter，日志会保存在 log_dir_base/experiment_name/timestamp 下。"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(log_dir_base, experiment_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    return writer


def get_optimizer(model, config_optimizer, learning_rate, weight_decay,
                  finetune_fc_lr=None, finetune_base_lr=None):
    """
    根据配置创建优化器。
    对于微调，可以为fc层和其余层设置不同学习率。
    """
    if finetune_fc_lr is not None and finetune_base_lr is not None:
        # 微调：不同学习率
        # 假设模型中最后一层命名为 'fc' (ResNet) 或 'classifier[6]' (AlexNet)
        # 或更通用的方式是找到最后一层
        if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
            fc_params = model.fc.parameters()
            fc_layer_name = "fc"
        elif (hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential)
              and isinstance(model.classifier[-1], torch.nn.Linear)):
            fc_params = model.classifier[-1].parameters()
            fc_layer_name = f"classifier.{len(model.classifier)-1}"  # e.g. classifier.6 for AlexNet
        else:  # Fallback: 尝试找到最后一个 nn.Linear 模块
            last_linear_name = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    last_linear_name = name
            if last_linear_name:
                fc_params = getattr(
                    model, last_linear_name).parameters() \
                    if '.' not in last_linear_name else model.get_submodule(last_linear_name).parameters()
                fc_layer_name = last_linear_name
                print(f"Auto-detected last linear layer as: {fc_layer_name}")
            else:
                raise ValueError(
                    "Could not automatically find the fully connected layer for differential learning rates.")

        base_params = [
            param for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith(fc_layer_name)
        ]

        params_to_optimize = [
            {'params': base_params, 'lr': finetune_base_lr},
            {'params': fc_params, 'lr': finetune_fc_lr}
        ]
        print(f"Optimizer: Differential LR. Base LR: {finetune_base_lr}, FC LR: {finetune_fc_lr}")
    else:
        # 从零训练或微调时所有层相同学习率
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        print(f"Optimizer: Uniform LR: {learning_rate}")

    if config_optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    elif config_optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif config_optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {config_optimizer} not supported.")
    return optimizer
