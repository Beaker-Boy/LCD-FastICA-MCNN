import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from mcnn_model import MSASCnn

def train_model(folder_path, train_mode, existing_model_path, new_model_path):
    """
    训练模型的主函数。
    假设数据已经由 build_tensor.py 处理好并保存在 folder_path 中。
    """
    # 加载预处理后的张量数据
    train_data_path = os.path.join(folder_path, "train_tensor_data.pt")
    val_data_path = os.path.join(folder_path, "val_tensor_data.pt")
    train_labels_path = os.path.join(folder_path, "train_labels.pt")
    val_labels_path = os.path.join(folder_path, "val_labels.pt")

    train_tensor_data = torch.load(train_data_path)
    val_tensor_data = torch.load(val_data_path)
    train_labels = torch.load(train_labels_path)
    val_labels = torch.load(val_labels_path)

    print(f"Loaded training data with shape: {train_tensor_data.shape}")
    print(f"Loaded validation data with shape: {val_tensor_data.shape}")
    print(f"Loaded training labels with shape: {train_labels.shape}")
    print(f"Loaded validation labels with shape: {val_labels.shape}")

    # 初始化模型、损失函数和优化器
    num_channels = train_tensor_data.shape[1]
    # 动态确定类别数量
    unique_classes = torch.unique(train_labels)
    num_classes = len(unique_classes)
    
    # 创建标签映射字典，将原始标签映射到连续的范围 [0, num_classes-1]
    class_to_idx = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    print(f"Label mapping: {class_to_idx}")
    
    # 应用标签映射
    train_labels_mapped = train_labels.clone()
    val_labels_mapped = val_labels.clone()
    for original_label, mapped_idx in class_to_idx.items():
        train_labels_mapped[train_labels == original_label] = mapped_idx
        val_labels_mapped[val_labels == original_label] = mapped_idx
    
    # 使用映射后的标签重新创建数据集
    train_dataset = TensorDataset(train_tensor_data, train_labels_mapped)
    val_dataset = TensorDataset(val_tensor_data, val_labels_mapped)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if train_mode == '读取已有模型继续训练':
        model = MSASCnn(num_channels, num_classes)
        model.load_state_dict(torch.load(existing_model_path))
        print(f"Loaded existing model from {existing_model_path}")
    else:
        model = MSASCnn(num_channels, num_classes)
        print(f"Initialized new model for {num_classes} classes.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_dataloader)

        # 验证模型
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_running_loss += loss.item()
                
                # 计算准确率（使用映射后的标签）
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print("训练完成")

    # 保存训练好的模型和标签映射
    torch.save(model.state_dict(), new_model_path)
    print(f"Model saved to {new_model_path}")
    
    # 保存标签映射字典，用于后续推理
    mapping_path = new_model_path.replace('.pth', '_label_mapping.pt')
    torch.save(class_to_idx, mapping_path)
    print(f"Label mapping saved to {mapping_path}")
