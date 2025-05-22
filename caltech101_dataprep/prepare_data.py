# caltech101_dataprep/prepare_data.py

from . import data_utils
from . import config


def verify_dataloaders(train_loader_, val_loader_, test_loader_, idx_to_class_):
    """验证DataLoaders并打印一些样本信息。"""
    print("\nVerifying DataLoaders...")
    print(f"Number of training batches: {len(train_loader_)}")
    print(f"Number of validation batches: {len(val_loader_)}")
    print(f"Number of testing batches: {len(test_loader_)}")

    print("\nSample batch from train_loader:")
    try:
        sample_images, sample_labels = next(iter(train_loader_))
        print(f"Images batch shape: {sample_images.size()}")
        print(f"Labels batch shape: {sample_labels.size()}")
        print(f"Sample labels (indices): {sample_labels[:5].tolist()}")
        if idx_to_class_:
            print(f"Sample label names: {[idx_to_class_[l.item()] for l in sample_labels[:5]]}")
        print(f"Min pixel value: {sample_images.min():.4f}, Max pixel value: {sample_images.max():.4f}")
    except Exception as e:
        print(f"Error getting batch from train_loader: {e}")

    print("\nSample batch from val_loader:")
    try:
        sample_images_val, sample_labels_val = next(iter(val_loader_))
        print(f"Val Images batch shape: {sample_images_val.size()}")
        print(f"Val Labels batch shape: {sample_labels_val.size()}")
    except Exception as e:
        print(f"Error getting batch from val_loader: {e}")


if __name__ == '__main__':
    print("Starting Caltech-101 data preparation...")

    # 获取DataLoaders和类别信息
    train_loader, val_loader, test_loader, class_to_idx, num_classes, idx_to_class = data_utils.get_dataloaders()

    print(f"\nData preparation complete. Number of classes: {num_classes}")

    # 运行验证
    verify_dataloaders(train_loader, val_loader, test_loader, idx_to_class)

    print(f"\nTrain, validation, and test DataLoaders are ready.")
    print(f"Class mapping saved to: {config.CLASS_MAPPING_FILE}")
