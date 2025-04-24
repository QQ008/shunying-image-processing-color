"""
训练脚本：用于训练分割模型和颜色分类模型
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from src.color_extraction.segmentation_model import UNet, CyclistSegmentationModel
from src.color_extraction.color_extraction_model import ColorClassificationCNN
from src.color_extraction.data_processor import CyclistImageDataset, create_dataloader
from utils.logger import setup_logger
import config

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="训练骑行者颜色提取模型")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="data/training",
                        help="训练数据目录")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="批处理大小")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                        help="数据加载线程数")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="segmentation",
                        choices=["segmentation", "classification"],
                        help="要训练的模型类型")
    parser.add_argument("--pretrained", action="store_true",
                        help="是否使用预训练模型")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    
    # 其他参数
    parser.add_argument("--save_dir", type=str, default="models",
                        help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志目录")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="训练设备")
    
    return parser.parse_args()

def train_segmentation_model(args, train_loader, val_loader, logger):
    """
    训练分割模型
    
    Args:
        args: 命令行参数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        logger: 日志记录器
    """
    # 初始化模型
    model = UNet(n_channels=3, n_classes=4)
    
    # 移动模型到设备
    device = torch.device(args.device)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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
        patience=5, 
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_start = time.time()
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            # 移动数据到设备
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_time = time.time() - train_start
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_start = time.time()
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                # 移动数据到设备
                images = images.to(device)
                masks = masks.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_time = time.time() - val_start
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录日志
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f} ({train_time:.2f}s), "
                   f"Val Loss: {val_loss:.4f} ({val_time:.2f}s)")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, "segmentation_model_best.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存最佳模型到 {model_path}")
        
        # 保存最近的模型
        model_path = os.path.join(args.save_dir, "segmentation_model_latest.pth")
        torch.save(model.state_dict(), model_path)
    
    logger.info("训练完成！")

def train_color_classification_model(args, train_loader, val_loader, logger):
    """
    训练颜色分类模型
    
    Args:
        args: 命令行参数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        logger: 日志记录器
    """
    # 初始化模型
    model = ColorClassificationCNN(num_classes=len(config.COLOR_CLASSES))
    
    # 移动模型到设备
    device = torch.device(args.device)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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
        patience=5, 
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_start = time.time()
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            # 移动数据到设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels).item() / labels.size(0)
            train_loss += loss.item()
        
        # 计算平均训练损失和准确率
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_time = time.time() - train_start
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_start = time.time()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                # 移动数据到设备
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 计算准确率
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels).item() / labels.size(0)
                val_loss += loss.item()
        
        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_time = time.time() - val_start
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录日志
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_time:.2f}s), "
                   f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_time:.2f}s)")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, "color_classification_model_best.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存最佳模型到 {model_path}")
        
        # 保存最近的模型
        model_path = os.path.join(args.save_dir, "color_classification_model_latest.pth")
        torch.save(model.state_dict(), model_path)
    
    logger.info("训练完成！")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志记录器
    log_file = os.path.join(args.log_dir, f"train_{args.model_type}.log")
    logger = setup_logger("train", log_file)
    
    # 记录训练配置
    logger.info(f"训练配置: {vars(args)}")
    
    # 加载数据
    # 注意：在实际应用中，需要根据实际的数据结构实现相应的数据集和加载器
    # 这里只是一个示例框架
    
    # 记录开始训练
    logger.info(f"开始训练 {args.model_type} 模型...")
    
    # 根据模型类型选择训练函数
    if args.model_type == "segmentation":
        train_segmentation_model(args, train_loader, val_loader, logger)
    else:  # classification
        train_color_classification_model(args, train_loader, val_loader, logger)

if __name__ == "__main__":
    main() 