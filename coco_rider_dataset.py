#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import h5py

class CocoRiderDataset(Dataset):
    """
    加载COCO格式标注的骑行者数据集
    
    该类读取COCO格式的标注文件，并将其与图像或预先提取的特征向量关联起来，
    用于训练颜色预测MLP模型。
    """
    def __init__(self, 
                 coco_file, 
                 image_dir=None,
                 features_file=None,
                 features_format='pickle',
                 transform=None,
                 color_mapping=None):
        """
        初始化数据集
        
        参数:
            coco_file: COCO格式标注文件路径
            image_dir: 图像目录路径（如果直接使用图像）
            features_file: 预先提取的特征向量文件路径（如果使用预提取特征）
            features_format: 特征文件格式 ('pickle', 'h5', 'json')
            transform: 应用于图像的转换（如果直接使用图像）
            color_mapping: 颜色名称到索引的映射字典
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 加载COCO格式标注
        with open(coco_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
            
        # 提取图像信息和标注信息
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # 创建颜色映射（如果未提供）
        if color_mapping is None:
            # 从标注中收集所有颜色值
            all_colors = set()
            for ann in self.coco_data['annotations']:
                for color_field in ['helmet_color', 'jersey_color', 'shorts_color', 'shoes_color']:
                    if color_field in ann and ann[color_field]:
                        all_colors.add(ann[color_field])
            
            # 创建颜色到索引的映射
            self.color_mapping = {color: idx for idx, color in enumerate(sorted(all_colors))}
            # 添加"未知"类别
            if 'unknown' not in self.color_mapping:
                self.color_mapping['unknown'] = len(self.color_mapping)
        else:
            self.color_mapping = color_mapping
            
        print(f"颜色映射: {self.color_mapping}")
        self.num_classes = len(self.color_mapping)
        print(f"颜色类别数: {self.num_classes}")
        
        # 构建样本列表
        self.samples = []
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            img_info = self.images[img_id]
            img_file = img_info['file_name']
            
            # 提取颜色标签
            labels = {}
            for part, field in [
                ('helmet', 'helmet_color'), 
                ('jersey', 'jersey_color'), 
                ('shorts', 'shorts_color'),
                ('shoes', 'shoes_color')
            ]:
                # 获取颜色值，不存在时使用'unknown'
                color = ann.get(field, 'unknown')
                if not color:
                    color = 'unknown'
                    
                # 将颜色名称转换为类别索引
                label = self.color_mapping.get(color, self.color_mapping['unknown'])
                labels[part] = label
            
            self.samples.append({
                'image_file': img_file,
                'labels': labels
            })
        
        # 加载特征向量（如果提供）
        self.features = None
        if features_file:
            self.features = self._load_features(features_file, features_format)
            print(f"已加载特征向量，数量: {len(self.features)}")
            
    def _load_features(self, features_file, format):
        """
        加载预先提取的特征向量
        
        参数:
            features_file: 特征文件路径
            format: 特征文件格式
            
        返回:
            特征向量字典 {文件名: 特征向量}
        """
        if format == 'pickle':
            with open(features_file, 'rb') as f:
                return pickle.load(f)
                
        elif format == 'h5':
            features = {}
            with h5py.File(features_file, 'r') as f:
                for key in f.keys():
                    # 还原文件名（如果有特殊字符替换）
                    orig_key = key.replace('_', '/', 1)  # 简单替换，可能需要调整
                    features[orig_key] = f[key][()]
            return features
            
        elif format == 'json':
            with open(features_file, 'r') as f:
                data = json.load(f)
            return {k: np.array(v) for k, v in data.items()}
            
        else:
            raise ValueError(f"不支持的特征文件格式: {format}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        img_file = sample['image_file']
        labels = sample['labels']
        
        # 转换标签为张量
        label_tensors = {part: torch.tensor(label, dtype=torch.long) 
                         for part, label in labels.items()}
        
        # 如果已加载特征向量，直接返回特征向量和标签
        if self.features is not None:
            # 检查特征向量中是否有该图像
            if img_file in self.features:
                features = torch.tensor(self.features[img_file], dtype=torch.float32)
                return features, label_tensors
            else:
                raise KeyError(f"在特征文件中未找到图像 '{img_file}'")
        
        # 否则，加载图像并应用转换
        if self.image_dir:
            img_path = os.path.join(self.image_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label_tensors
        
        raise ValueError("必须提供预提取的特征向量或图像目录")
    
    def get_color_name(self, idx):
        """根据索引获取颜色名称"""
        for color, color_idx in self.color_mapping.items():
            if color_idx == idx:
                return color
        return "unknown"
    
    @staticmethod
    def collate_fn(batch):
        """
        自定义批处理函数，用于处理不同长度的标签
        """
        # 分离特征/图像和标签
        features_or_images = [item[0] for item in batch]
        all_labels = [item[1] for item in batch]
        
        # 将特征/图像堆叠成批处理
        if torch.is_tensor(features_or_images[0]):
            features_or_images = torch.stack(features_or_images)
        
        # 合并标签
        batch_labels = {}
        for part in all_labels[0].keys():
            batch_labels[part] = torch.stack([labels[part] for labels in all_labels])
        
        return features_or_images, batch_labels

def save_color_mapping(color_mapping, output_file):
    """将颜色映射保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(color_mapping, f, ensure_ascii=False, indent=2)
    print(f"颜色映射已保存到 {output_file}")

# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试COCO骑行者数据集加载")
    parser.add_argument('--coco_file', type=str, required=True, help='COCO格式标注文件路径')
    parser.add_argument('--image_dir', type=str, default=None, help='图像目录路径')
    parser.add_argument('--features_file', type=str, default=None, help='特征向量文件路径')
    parser.add_argument('--features_format', type=str, default='pickle', choices=['pickle', 'h5', 'json'], help='特征文件格式')
    parser.add_argument('--save_mapping', type=str, default=None, help='保存颜色映射的文件路径')
    
    args = parser.parse_args()
    
    # 创建数据集
    dataset = CocoRiderDataset(
        coco_file=args.coco_file,
        image_dir=args.image_dir,
        features_file=args.features_file,
        features_format=args.features_format
    )
    
    # 打印数据集信息
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个样本: {dataset[0][1]}")  # 显示第一个样本的标签
    
    # 保存颜色映射（如果指定）
    if args.save_mapping:
        save_color_mapping(dataset.color_mapping, args.save_mapping)
    
    # 测试数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)
    
    # 获取一个批次
    x, y = next(iter(dataloader))
    print(f"批次形状: {x.shape}")
    print(f"标签: {y}") 