# cv-pj2-task1
Caltech-101 图像分类项目

## == 概述 ==
- 本项目使用PyTorch实现Caltech-101数据集的图像分类，支持对预训练模型（ResNet-18/AlexNet）进行微调或从头训练，包含超参数配置和实验跟踪功能。

主要特性：
- 支持训练集/验证集/测试集划分
- 可配置的训练参数（学习率、优化器等）
- 集成TensorBoard可视化训练指标
- 模型检查点保存与最佳模型选择
- 差异化的微调学习率设置

## == 项目结构 ==
```
caltech-101-classification/
├── caltech101_dataprep/          # 数据预处理模块
│   ├── __init__.py
│   ├── caltech101_class_to_idx.json  # 类别索引映射
│   ├── config.py                 # 数据路径配置
│   ├── data_utils.py             # 数据加载工具
│   ├── dataset.py                # Dataset类定义
│   ├── prepare_data.py           # 数据准备主脚本
│   └── transforms.py             # 数据增强策略
├── checkpoints/                  # 训练模型保存目录
├── hyperparams_search/           # 超参数搜索结果保存目录
├── model/                        # 模型定义
│   └── models.py                 # 模型架构实现
├── runs/                         # TensorBoard日志
├── utils/                        # 通用工具
│   ├── __init__.py
│   └── training_utils.py         # 训练工具
├── __init__.py
├── caltech101_class_to_idx.json  # 备用类别映射
├── config_train.py           # 训练参数配置
├── engine.py                 # 训练引擎
├── hyperparameter_sweep.py   # 超参数扫描脚本
├── main_finetune.py              # 微调训练入口
├── main_scratch.py               # 从头训练入口
└── README.md
```

## == 使用说明 ==
1. 准备数据集：
- 从官网下载Caltech-101数据集
- 修改`caltech101_dataprep/config.py `中的路径：
  DATA_ROOT_DIR = '您的数据集路径/101_ObjectCategories'

2. 运行训练：
- 微调预训练模型：
  `python caltech101_classification/main_finetune.py`
  
- 从头开始训练：
  `python caltech101_classification/main_scratch.py`

## == 配置调整 ==
修改 config_train.py 文件：
- MODEL_NAME: 选择"resnet18"或"alexnet"
- EPOCHS: 训练轮数（默认15）
- BATCH_SIZE: 根据显存调整（默认32）
- 学习率：微调层1e-5，新层1e-3
- 优化器：Adam/SGD可选

## == 训练监控 ==
使用TensorBoard查看指标：
tensorboard --logdir runs/

## == 注意事项 ==
1. 首次运行会自动下载预训练权重到D:/.../model_weights目录
2. 显存不足时可减小BATCH_SIZE值
3. 文档1中提供的实验配置可用于超参数搜索
4. 测试集评估需手动修改代码执行

许可证：MIT (详见LICENSE文件)
