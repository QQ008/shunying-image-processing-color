#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
import json

from color_predictor import ColorPredictorMLP, FeatureExtractor, RiderColorPredictor
from coco_rider_dataset import CocoRiderDataset, save_color_mapping

def train_one_epoch(model, dataloader, criterion, optimizer, device, feature_extractor=None):
    """训练模型一个周期"""
    model.train()
    running_loss = 0.0
    correct_preds = {part: 0 for part in model.color_predictor.part_names}
    total_preds = 0
    
    # 进度条
    pbar = tqdm(dataloader, desc="训练")
    
    for inputs, labels in pbar:
        # 将数据移动到设备
        inputs = inputs.to(device)
        labels = {part: label.to(device) for part, label in labels.items()}
        
        # 如果输入是图像而不是特征向量，先提取特征
        if feature_extractor is not None:
            with torch.no_grad():  # 不需要计算特征提取的梯度
                inputs = feature_extractor(inputs)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs) if feature_extractor is None else model.color_predictor(inputs)
        
        # 计算每个部分的损失并求和
        loss = 0
        for part in model.color_predictor.part_names:
            if part in labels:
                loss += criterion(outputs[part], labels[part])
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        total_preds += inputs.size(0)
        
        # 计算每个部分的准确率
        for part in model.color_predictor.part_names:
            if part in labels:
                pred_classes = torch.argmax(outputs[part], dim=1)
                correct_preds[part] += torch.sum(pred_classes == labels[part]).item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': sum(correct_preds.values()) / (total_preds * len(model.color_predictor.part_names))
        })
    
    # 计算周期统计
    epoch_loss = running_loss / total_preds
    epoch_acc = {part: correct / total_preds for part, correct in correct_preds.items()}
    avg_acc = sum(epoch_acc.values()) / len(epoch_acc)
    
    return epoch_loss, epoch_acc, avg_acc

def validate(model, dataloader, criterion, device, feature_extractor=None):
    """在验证集上验证模型"""
    model.eval()
    running_loss = 0.0
    correct_preds = {part: 0 for part in model.color_predictor.part_names}
    total_preds = 0
    
    # 不需要梯度计算
    with torch.no_grad():
        for inputs, labels in dataloader:
            # 将数据移动到设备
            inputs = inputs.to(device)
            labels = {part: label.to(device) for part, label in labels.items()}
            
            # 如果输入是图像而不是特征向量，先提取特征
            if feature_extractor is not None:
                inputs = feature_extractor(inputs)
            
            # 前向传播
            outputs = model(inputs) if feature_extractor is None else model.color_predictor(inputs)
            
            # 计算损失
            loss = 0
            for part in model.color_predictor.part_names:
                if part in labels:
                    loss += criterion(outputs[part], labels[part])
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            total_preds += inputs.size(0)
            
            # 计算每个部分的准确率
            for part in model.color_predictor.part_names:
                if part in labels:
                    pred_classes = torch.argmax(outputs[part], dim=1)
                    correct_preds[part] += torch.sum(pred_classes == labels[part]).item()
    
    # 计算验证统计
    val_loss = running_loss / total_preds
    val_acc = {part: correct / total_preds for part, correct in correct_preds.items()}
    avg_acc = sum(val_acc.values()) / len(val_acc)
    
    return val_loss, val_acc, avg_acc

def train_model(args):
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 加载已有的颜色映射（如果提供）
    color_mapping = None
    if args.color_mapping:
        with open(args.color_mapping, 'r', encoding='utf-8') as f:
            color_mapping = json.load(f)
        print(f"已加载颜色映射，类别数: {len(color_mapping)}")
    
    # 准备数据集
    transform = None
    if not args.features_file:
        # 如果使用原始图像，定义转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    # 创建数据集
    dataset = CocoRiderDataset(
        coco_file=args.coco_file,
        image_dir=args.image_dir,
        features_file=args.features_file,
        features_format=args.features_format,
        transform=transform,
        color_mapping=color_mapping
    )
    
    # 保存颜色映射（如果没有提供）
    if not args.color_mapping and args.save_mapping:
        save_color_mapping(dataset.color_mapping, args.save_mapping)
        print(f"已保存颜色映射到 {args.save_mapping}")
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    # 特征提取器和模型初始化
    feature_extractor = None
    
    # 确定特征维度
    if args.features_file:
        # 从第一个样本获取特征维度
        sample_features, _ = dataset[0]
        feature_dim = sample_features.shape[0]
        print(f"从预提取特征获取特征维度: {feature_dim}")
    else:
        # 使用指定的特征提取器
        feature_extractor = FeatureExtractor(
            model_name=args.backbone,
            pretrained=True,
            freeze_backbone=not args.finetune_backbone
        ).to(device)
        feature_dim = feature_extractor.feature_dim
        print(f"特征提取器特征维度: {feature_dim}")
    
    # 创建MLP模型
    mlp_model = ColorPredictorMLP(
        feature_dim=feature_dim,
        num_classes=dataset.num_classes,
        shared_hidden_dim=args.shared_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout_rate=args.dropout_rate,
        part_names=list(dataset.samples[0]['labels'].keys())  # 使用数据集中的部位名称
    ).to(device)
    
    # 如果使用单独的特征提取器
    if args.features_file:
        model = mlp_model
    else:
        # 创建完整模型（特征提取器 + MLP）
        model = RiderColorPredictor(
            feature_extractor_name=args.backbone,
            num_classes=dataset.num_classes,
            shared_hidden_dim=args.shared_hidden_dim,
            head_hidden_dim=args.head_hidden_dim,
            dropout_rate=args.dropout_rate,
            freeze_backbone=not args.finetune_backbone,
            part_names=list(dataset.samples[0]['labels'].keys())
        ).to(device)
        
        # 由于我们的验证使用单独的特征提取器，因此需要确保模型的MLP部分与单独的MLP模型相同
        mlp_model = model.color_predictor
    
    print(f"模型已创建，部位名称: {mlp_model.part_names}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 不同的学习率
    if args.features_file or not args.finetune_backbone:
        # 只训练MLP部分
        optimizer = optim.Adam(
            mlp_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        # 训练整个模型，但特征提取器使用较小的学习率
        backbone_params = list(model.feature_extractor.parameters())
        classifier_params = list(model.color_predictor.parameters())
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # 特征提取器较小的学习率
            {'params': classifier_params, 'lr': args.lr}       # MLP使用正常学习率
        ], weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # 初始化跟踪变量
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    print(f"开始训练 {args.epochs} 个周期...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练和验证
        train_loss, train_acc, train_avg_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            feature_extractor if args.features_file else None
        )
        val_loss, val_acc, val_avg_acc = validate(
            model, val_loader, criterion, device,
            feature_extractor if args.features_file else None
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存统计数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_avg_acc)
        val_accs.append(val_avg_acc)
        
        # 打印周期统计
        time_elapsed = time.time() - start_time
        print(f"周期 {epoch+1}/{args.epochs} 完成，用时 {time_elapsed:.2f}秒")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_avg_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_avg_acc:.4f}")
        
        # 打印每个部分的准确率
        print("各部位准确率:")
        for part, acc in val_acc.items():
            print(f"  - {part}: {acc:.4f}")
        
        # 如果验证损失改善，保存检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'color_mapping': dataset.color_mapping,
                'num_classes': dataset.num_classes,
                'part_names': mlp_model.part_names,
                'feature_dim': feature_dim,
                'shared_hidden_dim': args.shared_hidden_dim,
                'head_hidden_dim': args.head_hidden_dim,
                'dropout_rate': args.dropout_rate
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"已保存最佳模型检查点到 {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    
    # 保存检查点
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'color_mapping': dataset.color_mapping,
        'num_classes': dataset.num_classes,
        'part_names': mlp_model.part_names,
        'feature_dim': feature_dim,
        'shared_hidden_dim': args.shared_hidden_dim,
        'head_hidden_dim': args.head_hidden_dim,
        'dropout_rate': args.dropout_rate
    }
    
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"已保存最终模型到 {final_checkpoint_path}")
    
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
    print(f"已保存训练曲线到 {plot_path}")
    
    return model, mlp_model

def main():
    parser = argparse.ArgumentParser(description="使用COCO格式标注训练骑行者颜色预测模型")
    
    # 数据集参数
    parser.add_argument('--coco_file', type=str, required=True,
                       help='COCO格式标注文件路径')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图像目录路径（如果直接使用图像）')
    parser.add_argument('--features_file', type=str, default=None,
                       help='预提取特征向量文件路径')
    parser.add_argument('--features_format', type=str, default='pickle',
                       choices=['pickle', 'h5', 'json'],
                       help='特征文件格式')
    parser.add_argument('--color_mapping', type=str, default=None,
                       help='颜色映射文件路径（如果不提供，将从标注中自动生成）')
    parser.add_argument('--save_mapping', type=str, default=None,
                       help='保存颜色映射的文件路径')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='用于验证的数据比例')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small',
                       choices=['mobilenet_v3_small', 'mobilenet_v3_large'],
                       help='特征提取的backbone架构')
    parser.add_argument('--shared_hidden_dim', type=int, default=256,
                       help='共享隐藏层维度')
    parser.add_argument('--head_hidden_dim', type=int, default=128,
                       help='预测头隐藏层维度')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                       help='Dropout比率')
    parser.add_argument('--finetune_backbone', action='store_true',
                       help='是否微调backbone（仅当不使用预提取特征时有效）')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练周期数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载的工作线程数')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA，即使可用')
    
    # 输出参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='保存模型检查点的目录')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.features_file is None and args.image_dir is None:
        parser.error("必须提供--features_file或--image_dir之一")
    
    # 训练模型
    model, mlp_model = train_model(args)
    
    return model, mlp_model

if __name__ == '__main__':
    main() 