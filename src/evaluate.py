import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mcnn_model import MSASCnn
from cnn_models import create_model

# Configure matplotlib to use non-interactive backend
plt.switch_backend('Agg')


class ModelEvaluator:
    """模型评估器，提供完整的评估功能"""
    
    def __init__(self, model_path, data_dir, device=None):
        """
        初始化评估器
        
        Args:
            model_path: 模型权重文件路径 (.pth)
            data_dir: 数据集目录（包含 .pt 文件）
            device: 计算设备（CPU 或 CUDA）
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型和数据
        self._load_model_and_data()
        
        # 存储评估结果
        self.results = {}
    
    def _load_model_and_data(self):
        """加载模型和数据集"""
        print(f"使用设备：{self.device}")
        
        # 加载模型
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 从 checkpoint 中获取模型结构信息
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            model_info = checkpoint.get('model_info', {})
        else:
            # 向后兼容：直接是 state_dict
            model_state = checkpoint
            model_info = {}
        
        # 推断模型参数
        # 尝试从保存的信息中获取，或使用默认值
        num_channels = model_info.get('num_channels', 1)  # 默认 1 通道
        num_classes = model_info.get('num_classes', 10)   # 默认 10 类
        
        # 检测模型架构
        model_arch = model_info.get('model_arch', 'MCNN')  # 默认为 MCNN
        
        # 如果没有在 checkpoint 中保存架构信息，尝试从文件名或旁边的 .txt 文件读取
        if model_arch == 'MCNN':
            arch_info_path = self.model_path.replace('.pth', '_arch_info.txt')
            if os.path.exists(arch_info_path):
                with open(arch_info_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('Model Architecture:'):
                            model_arch = line.split(':')[1].strip()
                            break
        
        print(f"加载模型：架构={model_arch}, 通道数={num_channels}, 类别数={num_classes}")
        
        # 根据架构创建模型
        if model_arch == 'MCNN':
            self.model = MSASCnn(num_channels, num_classes)
        else:
            try:
                self.model = create_model(model_arch, num_channels, num_classes)
            except ValueError as e:
                print(f"警告：无法识别模型架构 '{model_arch}'，回退到 MCNN")
                self.model = MSASCnn(num_channels, num_classes)
        
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载数据
        test_data_path = os.path.join(self.data_dir, "test_tensor_data.pt")
        test_labels_path = os.path.join(self.data_dir, "test_labels.pt")
        
        # 如果没有测试集，使用验证集
        if not os.path.exists(test_data_path):
            val_data_path = os.path.join(self.data_dir, "val_tensor_data.pt")
            val_labels_path = os.path.join(self.data_dir, "val_labels.pt")
            
            if os.path.exists(val_data_path):
                test_data_path = val_data_path
                test_labels_path = val_labels_path
                print("未找到测试集，使用验证集进行评估")
            else:
                raise FileNotFoundError(f"在目录 {self.data_dir} 中未找到数据文件")
        
        self.test_data = torch.load(test_data_path).to(self.device)
        self.test_labels = torch.load(test_labels_path).to(self.device)
        
        print(f"加载测试数据：形状={self.test_data.shape}, 标签形状={self.test_labels.shape}")
        
        # 创建数据加载器
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        # 获取类别映射
        self.unique_classes = torch.unique(self.test_labels).tolist()
        self.num_classes = len(self.unique_classes)
        print(f"检测到 {self.num_classes} 个类别：{self.unique_classes}")
    
    def evaluate(self, save_dir=None):
        """
        执行完整评估
        
        Args:
            save_dir: 结果保存目录（可选）
        
        Returns:
            results: 评估结果字典
        """
        print("\n" + "="*60)
        print("开始模型评估")
        print("="*60)
        
        # 1. 进行预测
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n进行预测...")
        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                
                # 获取预测结果和概率
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)
        
        # 2. 计算各项指标
        print("\n计算评估指标...")
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        # 对于多分类，使用 macro/weighted 平均
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 详细分类报告
        report = classification_report(
            all_labels, 
            all_preds,
            target_names=[f'Class_{i}' for i in range(self.num_classes)],
            output_dict=True,
            zero_division=0
        )
        
        self.results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_preds.tolist(),
            'true_labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
            'model_path': self.model_path,
            'data_dir': self.data_dir,
            'num_samples': len(all_labels),
            'num_classes': self.num_classes,
            'timestamp': datetime.now().isoformat()
        }
        
        # 3. 打印结果
        print("\n" + "-"*60)
        print("评估结果汇总")
        print("-"*60)
        print(f"准确率 (Accuracy):     {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率 (Precision-Macro):    {precision_macro:.4f}")
        print(f"精确率 (Precision-Weighted): {precision_weighted:.4f}")
        print(f"召回率 (Recall-Macro):       {recall_macro:.4f}")
        print(f"召回率 (Recall-Weighted):    {recall_weighted:.4f}")
        print(f"F1 分数 (F1-Macro):          {f1_macro:.4f}")
        print(f"F1 分数 (F1-Weighted):       {f1_weighted:.4f}")
        print(f"样本总数：{len(all_labels)}")
        print("-"*60)
        
        # 打印详细分类报告
        print("\n详细分类报告:")
        print(classification_report(
            all_labels, 
            all_preds,
            target_names=[f'Class_{i}' for i in range(self.num_classes)],
            zero_division=0
        ))
        
        # 4. 保存结果
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_results(save_dir, all_preds, all_labels, all_probs, cm)
            print(f"\n评估结果已保存至：{save_dir}")
        
        return self.results
    
    def _save_results(self, save_dir, predictions, true_labels, probabilities, confusion_mat):
        """保存评估结果"""
        # 保存 JSON 结果
        json_path = os.path.join(save_dir, "evaluation_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 绘制并保存混淆矩阵
        self._plot_confusion_matrix(confusion_mat, save_dir)
        
        # 绘制并保存 t-SNE 图
        self._plot_tsne(predictions, true_labels, save_dir)
        
        # 保存预测结果对比
        comparison_df = {
            'sample_id': list(range(len(true_labels))),
            'true_label': true_labels,
            'predicted_label': predictions,
            'correct': (true_labels == predictions).astype(int)
        }
        import pandas as pd
        df = pd.DataFrame(comparison_df)
        csv_path = os.path.join(save_dir, "predictions_comparison.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存每个类别的概率分布
        prob_stats = []
        for i in range(self.num_classes):
            class_mask = true_labels == i
            if class_mask.sum() > 0:
                correct_probs = probabilities[class_mask & (predictions == i), i]
                wrong_probs = probabilities[class_mask & (predictions != i), i]
                prob_stats.append({
                    'class': i,
                    'samples': int(class_mask.sum()),
                    'correct_predictions': int((class_mask & (predictions == i)).sum()),
                    'avg_correct_prob': float(correct_probs.mean()) if len(correct_probs) > 0 else 0,
                    'avg_wrong_prob': float(wrong_probs.mean()) if len(wrong_probs) > 0 else 0
                })
        
        prob_json_path = os.path.join(save_dir, "probability_statistics.json")
        with open(prob_json_path, 'w', encoding='utf-8') as f:
            json.dump(prob_stats, f, indent=2)
    
    def _plot_confusion_matrix(self, cm, save_dir):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        # 归一化混淆矩阵（按行）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 处理可能的除零问题
        cm_normalized = np.nan_to_num(cm_normalized)
        
        plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Normalized Rate')
        
        # 添加标题和标签
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, [str(i) for i in range(self.num_classes)], rotation=45)
        plt.yticks(tick_marks, [str(i) for i in range(self.num_classes)])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Normalized)')
        
        # 在每个格子中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        
        plt.tight_layout()
        
        # 保存图片
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 混淆矩阵已保存：{cm_path}")
    
    def _plot_tsne(self, predictions, true_labels, save_dir):
        """绘制 t-SNE 特征分布图"""
        print("\n生成 t-SNE 可视化...")
        
        # 提取特征（使用全局平均池化后的特征）
        features = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                batch_data = batch_data.to(self.device)
                
                # 提取中间层特征
                x = self.model.wide_conv(batch_data)
                x = self.model.pool1(x)
                x = self.model.msasc_block1(x)
                x = self.model.msasc_block2(x)
                x = self.model.pool2(x)
                x = self.model.msasc_block3(x)
                x = self.model.global_avg_pool(x)
                x = x.view(x.size(0), -1)  # 展平
                
                features.append(x.cpu().numpy())
                labels_list.append(batch_labels.numpy())
        
        features = np.vstack(features)
        labels_list = np.concatenate(labels_list)
        
        # 应用 t-SNE
        print("应用 t-SNE 降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//10))
        features_2d = tsne.fit_transform(features)
        
        # 绘制 t-SNE 图
        plt.figure(figsize=(12, 10))
        
        # 为每个类别分配不同颜色
        unique_labels = np.unique(labels_list)
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_list == label
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.6,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )
        
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('t-SNE Visualization of Learned Features', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        tsne_path = os.path.join(save_dir, "tsne_visualization.png")
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ t-SNE 可视化已保存：{tsne_path}")


def evaluate_single_model(model_path, data_dir, save_dir=None):
    """
    评估单个模型
    
    Args:
        model_path: 模型路径
        data_dir: 数据目录
        save_dir: 结果保存目录
    
    Returns:
        results: 评估结果
    """
    evaluator = ModelEvaluator(model_path, data_dir)
    results = evaluator.evaluate(save_dir)
    return results


def compare_models(model_paths, data_dir, output_dir):
    """
    对比多个模型
    
    Args:
        model_paths: 模型路径列表
        data_dir: 数据目录
        output_dir: 输出目录
    
    Returns:
        comparison_results: 对比结果
    """
    print("\n" + "="*60)
    print("开始模型对比实验")
    print("="*60)
    
    comparison_results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n评估模型：{model_name}")
        
        # 为每个模型创建独立的子目录
        model_save_dir = os.path.join(output_dir, os.path.splitext(model_name)[0])
        
        try:
            results = evaluate_single_model(model_path, data_dir, model_save_dir)
            
            comparison_results.append({
                'model_name': model_name,
                'model_path': model_path,
                'accuracy': results['accuracy'],
                'f1_macro': results['f1_macro'],
                'f1_weighted': results['f1_weighted'],
                'precision_macro': results['precision_macro'],
                'recall_macro': results['recall_macro'],
                'num_samples': results['num_samples']
            })
        except Exception as e:
            print(f"✗ 评估失败：{e}")
            comparison_results.append({
                'model_name': model_name,
                'model_path': model_path,
                'error': str(e)
            })
    
    # 保存对比结果
    comparison_json = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_json, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    # 打印对比表格
    print("\n" + "="*80)
    print("模型对比结果")
    print("="*80)
    print(f"{'模型名称':<40} {'准确率':<10} {'F1-Macro':<10} {'F1-Weighted':<12}")
    print("-"*80)
    
    for result in comparison_results:
        if 'error' not in result:
            print(f"{result['model_name']:<40} "
                  f"{result['accuracy']:.4f}     "
                  f"{result['f1_macro']:.4f}     "
                  f"{result['f1_weighted']:.4f}")
    
    print("="*80)
    print(f"\n对比结果已保存：{comparison_json}")
    
    return comparison_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--save_dir", type=str, default="results/evaluation", help="结果保存目录")
    parser.add_argument("--compare", nargs='+', help="对比多个模型（传入多个模型路径）")
    
    args = parser.parse_args()
    
    if args.compare:
        # 多模型对比模式
        model_paths = [args.model_path] + args.compare
        compare_models(model_paths, args.data_dir, args.save_dir)
    else:
        # 单模型评估模式
        evaluate_single_model(args.model_path, args.data_dir, args.save_dir)