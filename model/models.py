# caltech101_classification/model/models.py
import os
import torchvision
import torch.nn as nn

# --- 设置自定义模型缓存目录 ---
# 重要：将下面的路径修改为你希望存储预训练权重的目录。
# 确保你对这个目录有写入权限。
# 例如: "D:/my_pytorch_models" 或 "/home/user/pytorch_models"
# 如果将其设置为空字符串或 None，PyTorch 将使用默认路径。
CUSTOM_MODEL_CACHE_DIR = r"D:\python object\computer vision\project2\task1\model_weights"

if CUSTOM_MODEL_CACHE_DIR:
    print(f"Attempting to set TORCH_HOME to: {CUSTOM_MODEL_CACHE_DIR}")
    os.environ['TORCH_HOME'] = CUSTOM_MODEL_CACHE_DIR
    # PyTorch Hub 会在 TORCH_HOME 下创建 'hub/checkpoints' 子目录来存放模型
    # 我们可以预先创建 TORCH_HOME 目录，以确保权限等问题
    try:
        # PyTorch 通常会在 $TORCH_HOME/hub/checkpoints 存储模型
        # 我们创建 $TORCH_HOME 目录，以及 $TORCH_HOME/hub，torch.hub 会负责 checkpoints
        os.makedirs(CUSTOM_MODEL_CACHE_DIR, exist_ok=True)
        # torch.hub 会在 $TORCH_HOME 下创建 'hub' 目录，如果它不存在
        # os.makedirs(os.path.join(CUSTOM_MODEL_CACHE_DIR, 'hub'), exist_ok=True)
        print(f"INFO: TORCH_HOME set to '{os.environ['TORCH_HOME']}'. Pretrained models will be cached here.")
        print(f"INFO: Expected model cache location: {os.path.join(os.environ['TORCH_HOME'], 'hub', 'checkpoints')}")
    except OSError as e:
        print(f"WARNING: Could not create TORCH_HOME directory '{CUSTOM_MODEL_CACHE_DIR}': {e}")
        print("WARNING: Please ensure the directory exists and is writable, or that you have permissions to create it.")
        print("WARNING: Model download might fail or use the default cache location.")
else:
    print("INFO: CUSTOM_MODEL_CACHE_DIR is not set. Using default PyTorch Hub cache location.")
# --- 结束自定义模型缓存目录设置 ---


def get_resnet18(num_classes, pretrained=True):
    """
    加载 ResNet-18 模型，可以带预训练权重，并修改最后一层以适应新的类别数。
    """
    if pretrained:
        print("Loading PRETRAINED ResNet-18 model.")
        # weights 参数会自动处理下载和缓存
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    else:
        print("Loading ResNet-18 model from SCRATCH (random weights).")
        model = torchvision.models.resnet18(weights=None)  # PyTorch 1.9+ 推荐使用 weights=None

    # 获取原始全连接层的输入特征数
    num_ftrs = model.fc.in_features

    # 替换为新的全连接层
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced ResNet-18's fc layer to output {num_classes} classes.")

    return model


def get_alexnet(num_classes, pretrained=True):
    """
    加载 AlexNet 模型，并修改分类器。
    """
    if pretrained:
        print("Loading PRETRAINED AlexNet model.")
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        model = torchvision.models.alexnet(weights=weights)
    else:
        print("Loading AlexNet model from SCRATCH (random weights).")
        model = torchvision.models.alexnet(weights=None)

    # AlexNet的分类器是 model.classifier[6]
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced AlexNet's classifier layer to output {num_classes} classes.")
    return model


# --- 统一的模型获取函数 ---
def get_model(model_name, num_classes, pretrained=True):
    # 确保 TORCH_HOME 设置在任何模型加载之前生效
    # (已移到文件顶部，所以这里不需要再次检查，但作为提醒)

    if model_name.lower() == "resnet18":
        return get_resnet18(num_classes, pretrained=pretrained)
    elif model_name.lower() == "alexnet":
        return get_alexnet(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported yet.")
