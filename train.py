 import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time

from color_predictor import RiderColorPredictor, ColorPredictorMLP, FeatureExtractor

# ----- 数据集类 -----

class RiderColorDataset(Dataset):
    """
    骑手颜色预测的数据集。
    
    这通常会加载裁剪后的骑手图像及其颜色标签。
    作为演示，这是一个占位符实现。
    """
    def __init__(self, 
                 image_dir: str,
                 label_file: str = None, 
                 transform=None,
                 part_names=None):
        """
        参数：
            image_dir: 包含骑手图像的目录
            label_file: 包含颜色标签的文件路径
            transform: 可选的图像转换
            part_names: 身体部位名称列表
        """
        self.image_dir = image_dir
        self.transform = transform
        self.part_names = part_names or ['helmet', 'jersey', 'shorts', 'shoes']
        
        # 加载数据
        # 这是一个占位符 - 在实际实现中，您会：
        # 1. 从image_dir读取图像
        # 2. 从label_file读取标签
        # 3. 创建图像和标签之间的映射
        
        # 为了演示目的，我们假设：
        self.image_files = [f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # 为演示创建虚拟标签
        # 在实际实现中，您会从label_file加载这些
        self.labels = {}
        for img_file in self.image_files:
            # 为每个部位生成随机标签（类别0-13）
            self.labels[img_file] = {
                part: np.random.randint(0, 14) for part in self.part_names
            }
            
        print(f"加载了包含 {len(self.image_files)} 张图像的数据集")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取图像及其标签"""
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换（如果有）
        if self.transform:
            image = self.transform(image)
        
        # 获取此图像的标签
        labels = self.labels[img_file]
        
        # 将标签转换为张量
        label_tensor = {part: torch.tensor(label, dtype=torch.long) 
                        for part, label in labels.items()}
        
        return image, label_tensor

# ----- 训练函数 -----

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练模型一个周期"""
    model.train()
    running_loss = 0.0
    correct_preds = {part: 0 for part in model.color_predictor.part_names}
    total_preds = 0
    
    # 进度条
    pbar = tqdm(dataloader, desc="训练中")
    
    for images, labels in pbar:
        # 将数据移动到设备
        images = images.to(device)
        labels = {part: label.to(device) for part, label in labels.items()}
        
        # 梯度归零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算每个部位的损失并求和
        loss = 0
        for part in model.color_predictor.part_names:
            loss += criterion(outputs[part], labels[part])
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计数据
        running_loss += loss.item() * images.size(0)
        total_preds += images.size(0)
        
        # 计算每个部位的准确率
        for part in model.color_predictor.part_names:
            pred_classes = torch.argmax(outputs[part], dim=1)
            correct_preds[part] += torch.sum(pred_classes == labels[part]).item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': sum(correct_preds.values()) / (total_preds * len(model.color_predictor.part_names))
        })
    
    # 计算周期统计数据
    epoch_loss = running_loss / total_preds
    epoch_acc = {part: correct / total_preds for part, correct in correct_preds.items()}
    avg_acc = sum(epoch_acc.values()) / len(epoch_acc)
    
    return epoch_loss, epoch_acc, avg_acc

def validate(model, dataloader, criterion, device):
    """在验证集上验证模型"""
    model.eval()
    running_loss = 0.0
    correct_preds = {part: 0 for part in model.color_predictor.part_names}
    total_preds = 0
    
    # 验证不需要梯度
    with torch.no_grad():
        for images, labels in dataloader:
            # 将数据移动到设备
            images = images.to(device)
            labels = {part: label.to(device) for part, label in labels.items()}
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = 0
            for part in model.color_predictor.part_names:
                loss += criterion(outputs[part], labels[part])
            
            # 统计数据
            running_loss += loss.item() * images.size(0)
            total_preds += images.size(0)
            
            # 计算每个部位的准确率
            for part in model.color_predictor.part_names:
                pred_classes = torch.argmax(outputs[part], dim=1)
                correct_preds[part] += torch.sum(pred_classes == labels[part]).item()
    
    # 计算验证统计数据
    val_loss = running_loss / total_preds
    val_acc = {part: correct / total_preds for part, correct in correct_preds.items()}
    avg_acc = sum(val_acc.values()) / len(val_acc)
    
    return val_loss, val_acc, avg_acc

# ----- 主训练循环 -----

def train_model(args):
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = RiderColorDataset(
        image_dir=args.data_dir,
        transform=transform,
        part_names=args.part_names
    )
    
    # 分割为训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = RiderColorPredictor(
        feature_extractor_name=args.backbone,
        num_classes=args.num_classes,
        shared_hidden_dim=args.shared_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout_rate=args.dropout_rate,
        freeze_backbone=not args.finetune_backbone,
        part_names=args.part_names
    )
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 如果微调，则骨干网络和分类器使用不同的学习率
    if args.finetune_backbone:
        # 新构造的模块的参数默认为requires_grad=True
        backbone_params = list(model.feature_extractor.parameters())
        classifier_params = list(model.color_predictor.parameters())
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # 预训练的骨干网络使用较低的学习率
            {'params': classifier_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # 初始化追踪变量
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 如果检查点目录不存在则创建
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    print(f"开始训练 {args.epochs} 个周期...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练和验证
        train_loss, train_acc, train_avg_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_avg_acc = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 存储统计数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_avg_acc)
        val_accs.append(val_avg_acc)
        
        # 打印周期统计数据
        time_elapsed = time.time() - start_time
        print(f"周期 {epoch+1}/{args.epochs} 完成，用时 {time_elapsed:.2f}秒")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_avg_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_avg_acc:.4f}")
        
        # 打印各部位准确率
        print("各部位准确率:")
        for part, acc in val_acc.items():
            print(f"  - {part}: {acc:.4f}")
        
        # 如果验证损失改善则保存检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"已将最佳模型检查点保存到 {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }, final_checkpoint_path)
    print(f"已将最终模型保存到 {final_checkpoint_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练')
    plt.plot(val_losses, label='验证')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练')
    plt.plot(val_accs, label='验证')
    plt.xlabel('周期')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
    plt.savefig(plot_path)
    print(f"已将训练曲线保存到 {plot_path}")
    
    return model

# ----- 主函数 -----

def main():
    parser = argparse.ArgumentParser(description="训练骑手颜色预测器模型")
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='包含骑手图像的目录')
    parser.add_argument('--part_names', type=str, nargs='+',
                        default=['helmet', 'jersey', 'shorts', 'shoes'],
                        help='要预测颜色的身体部位名称')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='用于验证的数据比例')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='特征提取的骨干架构')
    parser.add_argument('--num_classes', type=int, default=14,
                        help='要预测的颜色类别数量')
    parser.add_argument('--shared_hidden_dim', type=int, default=256,
                        help='共享隐藏层的维度')
    parser.add_argument('--head_hidden_dim', type=int, default=128,
                        help='预测头隐藏层的维度')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout率')
    parser.add_argument('--finetune_backbone', action='store_true',
                        help='是否微调骨干网络')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练的批量大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练的周期数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='优化器的权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--no_cuda', action='store_true',
                        help='即使可用也禁用CUDA')
    
    # 输出参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='保存模型检查点的目录')
    
    args = parser.parse_args()
    
    # 训练模型
    model = train_model(args)
    
    return model

if __name__ == '__main__':
    main()