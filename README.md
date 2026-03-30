# 掘进机主轴承振动信号故障诊断

## 项目概述
这是一个基于振动信号的故障诊断系统。它通过一个 PyQt5 图形用户界面（GUI）接收用户的操作指令，依次执行 LCD-FastICA 信号处理、数据张量构建和 MCNN 模型训练，最终生成一个能够根据振动信号判断设备故障类型的深度学习模型。

## 技术栈

1. **GUI框架**：PyQt5
2. **核心算法库**：Python, PyTorch, scikit-learn, NumPy, SciPy
3. **数据读写库**：NumPy (用于读取 NPY), SciPy (用于读写 MAT)
4. **数据存储格式**:
   - **原始数据**: `.npy`
   - **中间特征**: `.mat`
   - **模型输入**: `.pt` (PyTorch Tensor)

## 数据模型
由于项目不涉及传统关系型数据库，本系统的数据模型主要体现在多维数组（Tensor/Array）的结构与流转：
1. **原始数据 (Raw Data)**：从 `.npy` 文件中读取一维振动信号时间序列。
2. **中间特征 (Intermediate Features)**：经过 LCD-FastICA 处理后，生成的独立分量被保存为 `.mat` 文件。每个文件包含一个形状为 `(signal_length, num_components)` 的数组。
3. **模型输入数据 (Input Tensors)**：将多个 `.mat` 文件中的特征组合、切片、增广成多个样本，形成形状为 `(N_samples, N_channels, sample_length)` 的张量，并保存为 `.pt` 文件供模型直接读取。
4. **标签数据 (Labels)**：根据用户在 GUI 中为每个 `.npy` 文件选择的标签（如 'A', 'B' 等）生成。在数据构建阶段，为该文件产生的所有样本赋予相同的标签，并最终保存为 `.pt` 文件。

## 核心功能模块
根据项目架构，核心模块分为以下五个部分：
1. **GUI 交互模块 (`main_window.py`)**：
   - 提供图形界面，引导用户选择原始 `.npy` 文件、输入采样率及最大采样点数、选择标签和训练模式，并支持将其加入批处理列表。
   - 作为总控制器，按顺序调用信号处理、张量构建和模型训练三大核心流程。
2. **信号处理与降噪模块 (`lcd_fastica.py`)**：
   - 对输入的原始信号进行低通滤波。
   - **LCD (局部特征尺度分解)**：将非平稳、非线性的复杂振动信号分解为若干个内禀尺度分量 (ISC)。
   - **FastICA (快速独立成分分析)**：对分解出的 ISC 分量进行盲源分离，提取出更具代表性的独立特征成分，并保存为 `.mat` 文件。
3. **数据读取与张量构建模块 (`build_tensor.py`)**：
   - 读取 `LCD-FastICA` 模块输出的多个 `.mat` 特征文件。
   - 从每个特征文件中通过随机索引截取的方式，提取大量固定长度的样本片段。
   - 将提取的样本转换为多通道张量格式 `(N_samples, N_channels, sample_length)`。
   - 按 80/20 的比例划分训练集和验证集，并与标签一同保存为 `.pt` 文件。
   - 仅基于训练集计算均值和标准差，对训练集和验证集进行安全的归一化处理。
4. **深度学习模型构建模块 (MCNN) (`mcnn_model.py`)**：
   - 构建一个多尺度自适应卷积神经网络 (MSASCnn)。
   - **宽卷积层 (WideConvLayer)**：使用 128 和 64 两种不同大小的卷积核并行提取特征。
   - **多尺度自适应卷积块 (MSASCblock)**：通过三个并行的卷积层进一步提取多尺度信息。
5. **模型训练与评估模块 (`train_model.py`)**：
   - 加载 `.pt` 格式的训练和验证数据集。
   - 定义适合多分类任务的损失函数（如 CrossEntropy）及优化算法。
   - 支持从零开始训练或加载已有模型继续训练。
   - 实现标准的训练-验证循环，并在每个 epoch 结束后打印 Loss 和 Accuracy。

## 项目目录结构
```text
LCD-FastICA-MCNN/
├── data/               # 存放原始 .npy 信号文件
├── processed_data/     # 存放处理后的中间数据
│   ├── ica_results/    # 存放 LCD-FastICA 生成的 .mat 特征文件
│   └── tensor_dataset/ # 存放构建的 .pt 张量数据集与标签
├── models/             # 存放训练好的模型权重文件 (.pth)
├── src/                # 核心源代码目录
│   ├── main_window.py      # GUI 主程序
│   ├── lcd_fastica.py      # LCD 与 FastICA 信号处理实现
│   ├── build_tensor.py     # 张量构建与数据划分脚本
│   ├── mcnn_model.py       # MCNN 神经网络结构定义
│   └── train_model.py      # 模型训练脚本
├── requirements.txt    # Python 依赖包列表
└── README.md         # 项目文档
```