# 掘进机主轴承振动信号故障诊断系统 (LCD-FastICA-MCNN)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 **LCD-FastICA** 特征提取和 **多尺度卷积神经网络(MCNN)** 的旋转机械故障诊断系统，专为掘进机主轴承等工业设备设计。

## ✨ 核心特性

- 🎯 **核心算法**: LCD（局部特征尺度分解）+ FastICA（快速独立成分分析）组合算法
- 🧠 **多模型支持**: MCNN、WDCNN、ResCNN、SimpleCNN 四种深度学习架构
- 📊 **基线对比**: 内置 VMD、EEMD、LMD 等方法，便于性能对比实验
- 🤖 **智能优化**: PSO 粒子群优化自动调整信号处理参数
- 🖥️ **图形界面**: PyQt5 交互式 GUI，无需编程基础
- 📈 **全面评估**: 混淆矩阵、t-SNE 可视化、多指标分类报告

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy scipy matplotlib scikit-learn tqdm nptdms PyQt5 torch
pip install vmdpy PyEMD PyLMD  # 可选，用于基线对比
```

### 启动 GUI

```bash
python src/main_window.py
```

### 代码调用示例

```python
from src.lcd_fastica import process_signal_pipeline
from src.build_tensor import build_tensor_data
from src.train_model import train_model

# 1. 信号处理与特征提取
process_signal_pipeline(
    file_path='data/signal.npy',
    output_path='processed_data/features.mat',
    sampling_rate=20000,
    processing_methods=['LCD', 'FastICA']
)

# 2. 构建数据集
build_tensor_data(
    mat_files_dir='processed_data/',
    output_dir='processed_data/tensor_dataset/',
    sample_length=1024
)

# 3. 训练模型
train_model(
    data_dir='processed_data/tensor_dataset/',
    model_save_path='models/best_model.pth',
    epochs=50,
    batch_size=32
)
```

## 📂 项目结构

```
LCD-FastICA-MCNN/
├── src/                    # 源代码
│   ├── main_window.py     # GUI 主程序
│   ├── lcd_fastica.py     # 信号处理核心（LCD + FastICA）
│   ├── pso_optimizer.py   # PSO 参数优化器
│   ├── cnn_models.py      # 神经网络模型定义
│   ├── build_tensor.py    # 张量数据构建
│   ├── train_model.py     # 模型训练
│   └── evaluate.py        # 模型评估模块
├── data/                   # 原始振动信号数据 (.npy)
├── processed_data/         # 处理后的特征数据 (.mat, .pt)
├── models/                 # 训练好的模型权重 (.pth)
├── results/                # 评估结果与可视化图表
└── requirements.txt        # Python 依赖包
```

## 📖 主要功能模块

| 模块 | 功能说明 |
|------|---------|
| **信号处理** | LCD/VMD/EEMD/LMD 分解 + FastICA 盲源分离 |
| **参数优化** | PSO 自动优化 LCD 插值参数和 FastICA 收敛阈值 |
| **模型训练** | 支持 4 种 CNN 架构，断点续训，学习率调度 |
| **模型评估** | 准确率/精确率/召回率/F1，混淆矩阵，t-SNE 可视化 |
| **批处理** | 多样本批量处理，支持对比实验配置 |

## 🔬 典型应用场景

### 学术研究
- 对比 LCD-FastICA 与 VMD/EEMD/LMD 的性能差异
- 消融实验验证 FastICA 对特征可分性的提升
- 不同信噪比下的鲁棒性测试

### 工程应用
- 轴承早期故障检测
- 复合故障诊断
- 在线监测与预警

## 📝 更新日志

### v2.1 (2026-04-14)
- ✅ 新增中间产物图线输出功能
- ✅ PSO 参数优化集成到信号处理管道
- ✅ 支持 4 种 CNN 模型架构

### v2.0 (2026-03-31)
- ✅ 重构信号处理管道，简化方法组合逻辑
- ✅ 添加依赖预检和进度回调机制
- ✅ 完善异常处理和日志系统

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为此项目做出贡献的开发者和研究人员！

---

**文档最后更新**: 2026-04-14

# LCD-FastICA-MCNN 旋转机械故障诊断系统

基于局部特征尺度分解(LCD)和快速独立成分分析(FastICA)的多通道卷积神经网络(MCNN)旋转机械故障诊断系统。

## 🌟 核心特性

- 📊 **多传感器数据融合** - 支持振动、声音、温度等多类型传感器数据
- 🔧 **信号预处理** - LCD、VMD、EEMD、LMD等先进信号分解方法
- 🧠 **盲源分离** - FastICA算法提取故障特征
- 🧠 **深度学习** - MCNN/WDCNN/ResCNN/SimpleCNN多种神经网络架构
- 🤖 **参数优化** - 粒子群优化(PSO)自动优化LCD和FastICA参数
- 📈 **可视化分析** - 混淆矩阵、t-SNE特征分布、中间产物图线
- 🖥️ **图形界面** - 直观易用的操作界面
- 🧪 **全面评估** - 多指标模型性能评估

## 🧠 神经网络模型架构

| 模型 | 结构 | 参数量 | 训练速度 | 准确率 | 推荐场景 |
|------|------|--------|---------|--------|---------|
| **MCNN** | 多尺度自适应卷积 | 7.7M | 慢 | 最高 | 高精度研究 |
| **SimpleCNN** | 3层卷积+2层FC | 62K | 最快 | 中等 | 快速验证 |
| **WDCNN** | 宽深度网络 | 124K | 快 | 高 | 轴承诊断⭐ |
| **ResCNN** | 4残差块 | 1.1M | 中等 | 高 | 深层网络 |

## 🧪 PSO参数优化功能

使用粒子群优化(PSO)算法自动优化LCD和FastICA的关键参数，提升故障诊断准确性。

### 优化参数
- **LCD插值参数 a** ∈ [0.5, 2.0] - 控制基线构建的平滑度
- **FastICA收敛阈值 tol** ∈ [10⁻⁶, 10⁻³] - 控制ICA算法收敛精度

### 优化目标
最大化故障特征频率在信号中的幅值占比：
```
F = ΣA_i / (ΣA_i + ΣB_j)
```
其中 A_i 为故障特征频率幅值，B_j 为干扰频率幅值。

### 使用方法
1. 在GUI中勾选"启用PSO优化"复选框
2. 输入故障特征频率（多值，逗号分隔）
3. 可选调整PSO参数（粒子数、迭代次数等）

### 便捷函数
```python
from src.lcd_fastica import process_signal_pipeline_with_pso

signal, info = process_signal_pipeline_with_pso(
    file_path='data/signal.npy',
    output_path='results/output.mat',
    sampling_rate=20000,
    processing_methods=['LCD', 'FastICA'],
    fault_frequencies=[50.0, 100.0, 150.0],
    num_components=10
)
```

## 🛠️ 安装与使用

### 环境要求
- Python 3.7+
- PyTorch 1.7+
- Scikit-learn
- Matplotlib
- NumPy, SciPy

### 安装依赖
```bash
pip install -r requirements.txt
```

### 主要模块
- [src/lcd_fastica.py](file://e:\Github\LCD-FastICA-MCNN\src\lcd_fastica.py) - LCD和FastICA信号处理
- [src/pso_optimizer.py](file://e:\Github\LCD-FastICA-MCNN\src\pso_optimizer.py) - PSO参数优化器
- [src/cnn_models.py](file://e:\Github\LCD-FastICA-MCNN\src\cnn_models.py) - 多种CNN模型定义
- [src/train_model.py](file://e:\Github\LCD-FastICA-MCNN\src\train_model.py) - 模型训练逻辑
- [src/evaluate.py](file://e:\Github\LCD-FastICA-MCNN\src\evaluate.py) - 模型评估模块
- [src/main_window.py](file://e:\Github\LCD-FastICA-MCNN\src\main_window.py) - GUI主窗口

## 📊 信号处理流程

1. **数据预处理** - 低通滤波、归一化
2. **信号分解** - LCD/VMD/EEMD/LMD
3. **参数优化** - PSO算法优化参数（可选）
4. **盲源分离** - FastICA提取独立成分
5. **特征提取** - 时域、频域、时频域特征
6. **深度学习** - CNN模型训练与预测
7. **结果可视化** - 混淆矩阵、特征分布图等

## 📈 性能指标

- **准确率** - 预测正确的样本比例
- **精确率** - 预测为正例中实际为正例的比例
- **召回率** - 实际正例中预测正确的比例
- **F1分数** - 精确率和召回率的调和平均

## 📄 许可证

MIT License
