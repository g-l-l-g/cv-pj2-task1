# caltech101_dataprep/dataset.py
from PIL import Image
from torch.utils.data import Dataset
from . import config  # 用于 IMAGE_SIZE (如果占位符图像需要)


class Caltech101Dataset(Dataset):
    """自定义Caltech-101数据集类."""
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx  # 可选，用于调试或获取类别名

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')  # 确保是RGB
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 简单错误处理: 返回一个占位符图像和无效标签
            # 实际应用中可能需要更健壮的策略
            if len(self.image_paths) > 0 and idx != 0:
                # 尝试返回第一个样本，避免无限递归如果第一个也损坏
                return self.__getitem__(0)
            else:
                placeholder_img = Image.new('RGB', config.IMAGE_SIZE, (0, 0, 0))
                if self.transform:
                    placeholder_img = self.transform(placeholder_img)
                return placeholder_img, -1  # 返回一个无效标签

        if self.transform:
            image = self.transform(image)
        return image, label
