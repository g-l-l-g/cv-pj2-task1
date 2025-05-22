# caltech101_dataprep/data_utils.py

import os
import random
import json
import torch
from torch.utils.data import DataLoader
from . import config
from .dataset import Caltech101Dataset
from .transforms import get_train_transforms, get_val_test_transforms


def set_seed(seed_value):
    """设置随机种子以保证可复现性."""
    random.seed(seed_value)
    torch.manual_seed(seed_value)

    # 若cuda可用，则使用cuda加快运算
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # numpy.random.seed(seed_value) # 如果使用了numpy的随机操作


def scan_dataset(data_root_dir, exclude_dirs=None):
    """
    扫描数据集目录，收集所有图像路径和标签，并创建类别到索引的映射。
    返回:
        all_image_paths (list): 所有有效图片的路径列表。
        all_labels (list): 对应图片的标签索引列表。
        class_to_idx (dict): 类别名到整数索引的映射。
        idx_to_class (dict): 整数索引到类别名的映射。
        num_classes (int): 类别总数。
    """
    if exclude_dirs is None:
        exclude_dirs = []

    all_image_paths = []
    all_labels = []
    class_to_idx = {}
    idx_to_class = {}
    current_class_idx = 0

    print(f"Scanning dataset directory: {data_root_dir}")
    if not os.path.isdir(data_root_dir):
        raise FileNotFoundError(
            f"Dataset directory {data_root_dir} not found. "
            "Please check the path in config.py."
        )

    for class_name in sorted(os.listdir(data_root_dir)):
        class_dir = os.path.join(data_root_dir, class_name)
        if class_name in exclude_dirs or not os.path.isdir(class_dir):
            print(f"Skipping: {class_name}")
            continue

        if class_name not in class_to_idx:
            class_to_idx[class_name] = current_class_idx
            idx_to_class[current_class_idx] = class_name
            current_class_idx += 1

        image_filenames = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # 注意：这里打乱每个类内部文件顺序，以便后续按数量取样时是随机的
        random.shuffle(image_filenames)

        for img_name in image_filenames:
            all_image_paths.append(os.path.join(class_dir, img_name))
            all_labels.append(class_to_idx[class_name])

    num_total_images = len(all_image_paths)
    num_classes = len(class_to_idx)
    print(f"Found {num_total_images} images in {num_classes} classes.")
    if num_classes != 101 and "BACKGROUND_Google" in exclude_dirs:  # 针对Caltech-101的特定检查
        print(f"Warning: Expected 101 object classes (excluding background), but found {num_classes}."
              f"Check your dataset structure and EXCLUDE_DIRS in config.py.")
    elif num_classes == 0:
        raise ValueError("No classes found. Please check DATA_ROOT_DIR and dataset structure.")

    return all_image_paths, all_labels, class_to_idx, idx_to_class, num_classes


def split_data_indices(all_labels, num_classes, idx_to_class,
                       num_train_per_class, num_val_per_class):
    """
    根据每类的样本数划分训练集、验证集和测试集的索引。
    返回:
        train_indices (list): 训练集样本在原始列表中的索引。
        val_indices (list): 验证集样本在原始列表中的索引。
        test_indices (list): 测试集样本在原始列表中的索引。
    """
    train_indices = []
    val_indices = []
    test_indices = []

    indices_per_class = [[] for _ in range(num_classes)]
    for i, label_idx in enumerate(all_labels):
        indices_per_class[label_idx].append(i)

    for class_idx in range(num_classes):
        class_specific_indices = indices_per_class[class_idx]
        # 已经在scan_dataset中对每个类的文件名列表打乱过，
        # 这里可以再次打乱基于全局索引的列表，如果需要的话，但通常不是必需的。
        # random.shuffle(class_specific_indices)

        n_total_class_samples = len(class_specific_indices)
        n_train = num_train_per_class
        n_val = num_val_per_class

        if n_total_class_samples < n_train + n_val:
            class_name_str = idx_to_class.get(class_idx, f"ClassIdx_{class_idx}")
            print(
                f"Warning: Class '{class_name_str}' has only {n_total_class_samples} samples, "
                f"less than requested train ({n_train}) + val ({n_val}). "
                "Adjusting counts for this class."
            )
            # 调整策略：优先满足训练，然后验证，其余测试
            actual_train = min(n_train, n_total_class_samples)
            remaining_for_val_test = n_total_class_samples - actual_train
            actual_val = min(n_val, remaining_for_val_test)
            actual_test = remaining_for_val_test - actual_val

            train_indices.extend(class_specific_indices[:actual_train])
            val_indices.extend(
                class_specific_indices[actual_train: actual_train + actual_val]
            )
            test_indices.extend(
                class_specific_indices[actual_train + actual_val: actual_train + actual_val + actual_test]
            )
        else:
            train_indices.extend(class_specific_indices[:n_train])
            val_indices.extend(class_specific_indices[n_train: n_train + n_val])
            test_indices.extend(class_specific_indices[n_train + n_val:])

    print(f"Total train samples: {len(train_indices)}")
    print(f"Total validation samples: {len(val_indices)}")
    print(f"Total test samples: {len(test_indices)}")
    return train_indices, val_indices, test_indices


def get_dataloaders():
    """
    主函数，执行数据扫描、划分，并创建DataLoaders。
    返回:
        train_loader (DataLoader)
        val_loader (DataLoader)
        test_loader (DataLoader)
        class_to_idx (dict)
        num_classes (int)
    """
    set_seed(config.RANDOM_SEED)

    all_image_paths, all_labels, class_to_idx, idx_to_class, num_classes = \
        scan_dataset(config.DATA_ROOT_DIR, config.EXCLUDE_DIRS)

    train_indices, val_indices, test_indices = split_data_indices(
        all_labels, num_classes, idx_to_class,
        config.NUM_TRAIN_PER_CLASS, config.NUM_VAL_PER_CLASS
    )

    # 根据索引提取对应的路径和标签
    train_image_paths = [all_image_paths[i] for i in train_indices]
    train_image_labels = [all_labels[i] for i in train_indices]
    val_image_paths = [all_image_paths[i] for i in val_indices]
    val_image_labels = [all_labels[i] for i in val_indices]
    test_image_paths = [all_image_paths[i] for i in test_indices]
    test_image_labels = [all_labels[i] for i in test_indices]

    # 获取转换
    train_transform = get_train_transforms()
    val_test_transform = get_val_test_transforms()

    # 创建Dataset实例
    train_dataset = Caltech101Dataset(
        train_image_paths, train_image_labels,
        transform=train_transform, class_to_idx=class_to_idx
    )
    val_dataset = Caltech101Dataset(
        val_image_paths, val_image_labels,
        transform=val_test_transform, class_to_idx=class_to_idx
    )
    test_dataset = Caltech101Dataset(
        test_image_paths, test_image_labels,
        transform=val_test_transform, class_to_idx=class_to_idx
    )

    # 创建DataLoader实例
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # 保存类别映射
    try:
        with open(config.CLASS_MAPPING_FILE, 'w') as f:
            json.dump(class_to_idx, f, indent=4)
        print(f"Saved class_to_idx mapping to {config.CLASS_MAPPING_FILE}")
    except IOError as e:
        print(f"Error saving class_to_idx mapping: {e}")

    return train_loader, val_loader, test_loader, class_to_idx, num_classes, idx_to_class
