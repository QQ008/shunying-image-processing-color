"""
颜色提取模型：用于从分割区域提取主要颜色
"""

import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.color_utils import extract_dominant_colors, classify_color
import config

class ColorExtractionModel:
    """
    颜色提取模型
    """
    def __init__(self, segmentation_model=None, device=None):
        """
        初始化颜色提取模型
        
        Args:
            segmentation_model: 分割模型实例
            device: 计算设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_model = segmentation_model
        self.color_threshold = config.COLOR_THRESHOLD
        self.min_area_ratio = config.MIN_AREA_RATIO
    
    def extract_colors(self, image_path):
        """
        从图像中提取颜色
        
        Args:
            image_path: 图像路径
            
        Returns:
            color_results: 颜色结果字典 {部位名称: [(颜色名称, 置信度), ...]}
        """
        # 分割图像
        if self.segmentation_model:
            original_image, masks = self.segmentation_model.segment_image(image_path)
        else:
            # 如果没有分割模型，直接读取图像并使用简单启发式方法划分区域
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            masks = self._create_heuristic_masks(original_image)
        
        # 提取每个部位的颜色
        color_results = {}
        for part_name, mask in masks.items():
            # 计算掩码区域比例
            mask_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            
            # 如果区域太小，跳过
            if mask_ratio < self.min_area_ratio:
                color_results[part_name] = []
                continue
            
            # 提取该区域的主要颜色
            colors, percentages = extract_dominant_colors(original_image, k=3, mask=mask)
            
            # 分类颜色
            classified_colors = []
            for color, percentage in zip(colors, percentages):
                color_name, confidence = classify_color(color)
                # 只保留置信度高于阈值的颜色
                if confidence > self.color_threshold:
                    classified_colors.append((color_name, confidence))
            
            color_results[part_name] = classified_colors
        
        return color_results
    
    def _create_heuristic_masks(self, image):
        """
        创建启发式掩码（当没有分割模型时）
        
        Args:
            image: 图像
            
        Returns:
            masks: 掩码字典
        """
        height, width = image.shape[:2]
        
        # 创建各部位的掩码
        helmet_mask = np.zeros((height, width), dtype=np.uint8)
        upper_body_mask = np.zeros((height, width), dtype=np.uint8)
        lower_body_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 简单的启发式分割：上1/5为头盔，中间2/5为上衣，下2/5为裤子
        helmet_height = height // 5
        upper_body_height = 2 * height // 5
        
        helmet_mask[:helmet_height, :] = 255
        upper_body_mask[helmet_height:helmet_height+upper_body_height, :] = 255
        lower_body_mask[helmet_height+upper_body_height:, :] = 255
        
        return {
            "helmet": helmet_mask,
            "upper_body": upper_body_mask,
            "lower_body": lower_body_mask
        }


class ColorClassificationCNN(nn.Module):
    """
    颜色分类CNN模型
    """
    def __init__(self, num_classes=len(config.COLOR_CLASSES)):
        """
        初始化颜色分类CNN
        
        Args:
            num_classes: 颜色类别数量
        """
        super(ColorClassificationCNN, self).__init__()
        
        # 使用ResNet18作为特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 残差块1
            self._make_residual_block(64, 64, 2),
            
            # 残差块2
            self._make_residual_block(64, 128, 2, stride=2),
            
            # 残差块3
            self._make_residual_block(128, 256, 2, stride=2),
            
            # 残差块4
            self._make_residual_block(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels, blocks, stride=1):
        """
        创建残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            blocks: 块数量
            stride: 步长
            
        Returns:
            block: 残差块
        """
        layers = []
        
        # 第一个残差单元可能需要调整维度
        layers.append(self._residual_unit(in_channels, out_channels, stride))
        
        # 添加剩余的残差单元
        for _ in range(1, blocks):
            layers.append(self._residual_unit(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _residual_unit(self, in_channels, out_channels, stride=1):
        """
        残差单元
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            
        Returns:
            unit: 残差单元
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            x: 颜色分类概率
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 