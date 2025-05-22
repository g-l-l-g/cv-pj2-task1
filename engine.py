# caltech101_classification/engine.py
import torch
import time
from tqdm import tqdm


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, writer, print_freq=50):
    """训练模型一个epoch。"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds==labels.data)
        total_samples += labels.size(0)

        if (i + 1) % print_freq == 0 or (i + 1) == len(data_loader):
            avg_loss_batch = running_loss / total_samples
            acc_batch = correct_predictions.double() / total_samples
            progress_bar.set_postfix({'loss': f'{avg_loss_batch:.4f}', 'acc': f'{acc_batch:.4f}'})

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    epoch_time = time.time() - start_time

    if writer:
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    print(f"Epoch {epoch+1} [Train] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_acc


def evaluate(model, criterion, data_loader, device, epoch, writer, phase="Val"):
    """在验证集或测试集上评估模型。"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [{phase}]", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            avg_loss_batch = running_loss / total_samples
            acc_batch = correct_predictions.double() / total_samples
            progress_bar.set_postfix({'loss': f'{avg_loss_batch:.4f}', 'acc': f'{acc_batch:.4f}'})

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    epoch_time = time.time() - start_time

    if writer and phase.lower() == "val":  # 通常只记录验证集的指标以供epoch间的比较
        writer.add_scalar('Loss/validation', epoch_loss, epoch)
        writer.add_scalar('Accuracy/validation', epoch_acc, epoch)

    print(f"Epoch {epoch+1} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_acc
