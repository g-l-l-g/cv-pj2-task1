# caltech101_dataprep/transforms.py
from torchvision import transforms
from . import config


def get_train_transforms():
    """获取训练集的图像转换."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])


def get_val_test_transforms():
    """获取验证集和测试集的图像转换."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])
