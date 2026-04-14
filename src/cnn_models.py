import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 基础 CNN (Simple CNN) - 轻量化架构
# ============================================================================
class SimpleCNN(nn.Module):
    """
    基础卷积神经网络 - 轻量化架构
    
    结构: 3层卷积层 (5×1卷积核) + 2层全连接层
    
    Args:
        in_channels: 输入通道数 (对应信号的特征通道数)
        num_classes: 分类类别数
    """
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        
        # 3层卷积层 (5×1 卷积核)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全局平均池化 (自适应，不受输入长度限制)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2层全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第1个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # 第2个卷积块
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # 第3个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 2. WDCNN (Wide Deep Convolutional Neural Network)
# ============================================================================
class WDCNN(nn.Module):
    """
    宽深度卷积神经网络 (WDCNN)
    
    结构: 
    - 第1层: 128×1 宽卷积核 (捕获低频全局特征)
    - 后续5层: 小卷积核 (3×1) 逐层提取高频局部细节
    - 总共6层卷积层 + 3层全连接层
    
    Args:
        in_channels: 输入通道数
        num_classes: 分类类别数
    """
    def __init__(self, in_channels, num_classes):
        super(WDCNN, self).__init__()
        
        # 第1层: 宽卷积核 (128×1) - 捕获低频全局特征
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=128, stride=2, padding=63)
        self.bn1 = nn.BatchNorm1d(16)
        
        # 第2-6层: 小卷积核 (3×1) - 逐层提取高频细节
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(128)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 3层全连接层
        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第1层: 宽卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # 第2层
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # 第3层
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # 第4层
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # 第5层
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        
        # 第6层
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================================
# 3. ResCNN (Residual Convolutional Neural Network)
# ============================================================================
class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)
    
    结构: 2个卷积层 + 跳跃连接
    y = F(x) + x
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小 (5 或 7)
        stride: 步长
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlock, self).__init__()
        
        padding = kernel_size // 2  # 保持尺寸不变
        
        # 第1个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第2个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 跳跃连接的投影层 (当输入输出通道数不一致时)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 跳跃连接
        residual = self.shortcut(residual)
        
        # 相加 + ReLU
        out += residual
        out = F.relu(out)
        
        return out


class ResCNN(nn.Module):
    """
    残差卷积神经网络 (ResCNN)
    
    结构: 
    - 初始卷积层 (7×1)
    - 4个残差块 (交替使用 5×1 和 7×1 卷积核)
    - 全局平均池化 + 全连接层
    
    Args:
        in_channels: 输入通道数
        num_classes: 分类类别数
    """
    def __init__(self, in_channels, num_classes):
        super(ResCNN, self).__init__()
        
        # 初始卷积层 (7×1)
        self.initial_conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn_initial = nn.BatchNorm1d(64)
        self.pool_initial = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 4个残差块
        # 残差块1: 64 -> 64 (5×1)
        self.res_block1 = ResidualBlock(64, 64, kernel_size=5, stride=1)
        
        # 残差块2: 64 -> 128 (7×1, 下采样)
        self.res_block2 = ResidualBlock(64, 128, kernel_size=7, stride=2)
        
        # 残差块3: 128 -> 128 (5×1)
        self.res_block3 = ResidualBlock(128, 128, kernel_size=5, stride=1)
        
        # 残差块4: 128 -> 256 (7×1, 下采样)
        self.res_block4 = ResidualBlock(128, 256, kernel_size=7, stride=2)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn_initial(self.initial_conv(x)))
        x = self.pool_initial(x)
        
        # 4个残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# 模型工厂函数 (用于动态创建模型)
# ============================================================================
def create_model(model_name, in_channels, num_classes):
    """
    根据模型名称创建对应的模型实例
    
    Args:
        model_name: 模型名称 ('SimpleCNN', 'WDCNN', 'ResCNN', 'MCNN')
        in_channels: 输入通道数
        num_classes: 分类类别数
    
    Returns:
        model: PyTorch 模型实例
    
    Raises:
        ValueError: 如果模型名称不支持
    """
    models_dict = {
        'SimpleCNN': SimpleCNN,
        'WDCNN': WDCNN,
        'ResCNN': ResCNN,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(models_dict.keys())}")
    
    return models_dict[model_name](in_channels, num_classes)


def get_available_models():
    """
    获取所有可用的模型名称列表
    
    Returns:
        list: 可用模型名称列表
    """
    return ['SimpleCNN', 'WDCNN', 'ResCNN']


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("测试新增的神经网络模型")
    print("=" * 80)
    
    # 模拟输入数据 (batch_size=2, channels=5, length=1024)
    dummy_input = torch.randn(2, 5, 1024)
    num_classes = 10
    
    # 测试 SimpleCNN
    print("\n1. 测试 SimpleCNN...")
    try:
        model_simple = SimpleCNN(in_channels=5, num_classes=num_classes)
        output_simple = model_simple(dummy_input)
        print(f"   ✓ SimpleCNN 创建成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output_simple.shape}")
        print(f"   参数量: {sum(p.numel() for p in model_simple.parameters()):,}")
    except Exception as e:
        print(f"   ✗ SimpleCNN 失败: {e}")
    
    # 测试 WDCNN
    print("\n2. 测试 WDCNN...")
    try:
        model_wdcnn = WDCNN(in_channels=5, num_classes=num_classes)
        output_wdcnn = model_wdcnn(dummy_input)
        print(f"   ✓ WDCNN 创建成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output_wdcnn.shape}")
        print(f"   参数量: {sum(p.numel() for p in model_wdcnn.parameters()):,}")
    except Exception as e:
        print(f"   ✗ WDCNN 失败: {e}")
    
    # 测试 ResCNN
    print("\n3. 测试 ResCNN...")
    try:
        model_rescnn = ResCNN(in_channels=5, num_classes=num_classes)
        output_rescnn = model_rescnn(dummy_input)
        print(f"   ✓ ResCNN 创建成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output_rescnn.shape}")
        print(f"   参数量: {sum(p.numel() for p in model_rescnn.parameters()):,}")
    except Exception as e:
        print(f"   ✗ ResCNN 失败: {e}")
    
    # 测试模型工厂函数
    print("\n4. 测试模型工厂函数...")
    try:
        available_models = get_available_models()
        print(f"   可用模型: {available_models}")
        
        for model_name in available_models:
            model = create_model(model_name, in_channels=5, num_classes=num_classes)
            output = model(dummy_input)
            print(f"   ✓ {model_name}: 输出形状 {output.shape}")
    except Exception as e:
        print(f"   ✗ 模型工厂函数失败: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
