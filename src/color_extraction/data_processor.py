"""
数据处理模块：负责加载和预处理图像数据
"""

import cv2
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import log_error

class CyclistImageDataset(Dataset):
    """
    骑行者图像数据集
    """
    def __init__(self, image_paths, transform=None, logger=None):
        """
        初始化数据集
        
        Args:
            image_paths: 图像路径列表
            transform: 图像变换
            logger: 日志记录器
        """
        self.image_paths = image_paths
        self.transform = transform or self._default_transform()
        self.logger = logger
        self.valid_indices = []
        
        # 预处理：验证所有图像文件
        for i, path in enumerate(self.image_paths):
            try:
                # 尝试打开图像以验证它是否有效
                img = Image.open(path)
                img.verify()
                self.valid_indices.append(i)
            except Exception as e:
                if self.logger:
                    log_error(self.logger, path, str(e))
    
    def __len__(self):
        """返回有效图像数量"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        获取数据集项
        
        Args:
            idx: 索引
            
        Returns:
            image: 处理后的图像
            path: 图像路径
        """
        # 获取有效索引
        real_idx = self.valid_indices[idx]
        image_path = self.image_paths[real_idx]
        
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, image_path
        except Exception as e:
            if self.logger:
                log_error(self.logger, image_path, f"读取图像失败: {str(e)}")
            # 返回占位符图像
            return torch.zeros((3, 224, 224)), image_path
    
    def _default_transform(self):
        """
        默认图像变换
        
        Returns:
            transform: 默认变换管道
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_dataloader(image_paths, batch_size=32, num_workers=4, logger=None):
    """
    创建数据加载器
    
    Args:
        image_paths: 图像路径列表
        batch_size: 批大小
        num_workers: 工作线程数
        logger: 日志记录器
        
    Returns:
        dataloader: 数据加载器
    """
    dataset = CyclistImageDataset(image_paths, logger=logger)
    
    # 如果没有有效图像，返回None
    if len(dataset) == 0:
        if logger:
            logger.warning("没有找到有效图像")
        return None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def preprocess_image(image_path, target_size=(224, 224)):
    """
    预处理单个图像
    
    Args:
        image_path: 图像路径
        target_size: 目标大小
        
    Returns:
        processed_image: 处理后的图像
        original_image: 原始图像
    """
    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整大小
    processed_image = cv2.resize(original_image, target_size)
    
    # 归一化
    processed_image = processed_image.astype(np.float32) / 255.0
    
    # 转换为RGB (OpenCV使用BGR)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # 转换为PyTorch张量格式: (C, H, W)
    processed_image = np.transpose(processed_image, (2, 0, 1))
    
    # 添加批量维度
    processed_image = np.expand_dims(processed_image, axis=0)
    
    return processed_image, original_image 