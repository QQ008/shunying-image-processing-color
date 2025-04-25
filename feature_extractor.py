#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import argparse
from PIL import Image
import json
from tqdm import tqdm
import pickle
import h5py
from pathlib import Path

from color_predictor import FeatureExtractor

class YoloSegFeatureExtractor:
    """
    从YOLO-seg分割的骑行者图片中提取特征向量。
    
    该类加载预处理后的图片（已经通过YOLO-seg分割），使用预训练的CNN模型提取特征向量，
    并将结果保存到文件中供MLP模型使用。
    """
    def __init__(self, 
                 model_name='mobilenet_v3_small',
                 device=None,
                 batch_size=32):
        """
        初始化特征提取器
        
        参数:
            model_name: 特征提取器的CNN模型名称 ('mobilenet_v3_small', 'mobilenet_v3_large')
            device: 运行设备 ('cuda' 或 'cpu')
            batch_size: 批处理大小
        """
        # 设置设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置批处理大小
        self.batch_size = batch_size
        
        # 初始化特征提取器
        self.extractor = FeatureExtractor(
            model_name=model_name,
            pretrained=True,
            freeze_backbone=True
        )
        self.extractor.to(self.device)
        self.extractor.eval()
        
        print(f"特征提取器已初始化，特征向量维度: {self.extractor.feature_dim}")
    
    def extract_single_image(self, image_path):
        """
        从单个图像提取特征向量
        
        参数:
            image_path: 图像文件路径
            
        返回:
            特征向量 (numpy数组)
        """
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.extractor.preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.extractor(image_tensor)
        
        # 将张量转换为numpy数组
        features_np = features.cpu().numpy()
        
        return features_np[0]  # 返回单个特征向量
    
    def extract_batch(self, image_paths):
        """
        批量提取多个图像的特征向量
        
        参数:
            image_paths: 图像文件路径列表
            
        返回:
            特征向量列表 (numpy数组)
        """
        batch_size = len(image_paths)
        
        # 准备批处理张量
        batch_tensor = torch.zeros(batch_size, 3, 224, 224)
        
        # 加载并预处理每个图像
        for i, path in enumerate(image_paths):
            image = Image.open(path).convert('RGB')
            batch_tensor[i] = self.extractor.preprocess(image)
        
        # 移动到设备并提取特征
        batch_tensor = batch_tensor.to(self.device)
        with torch.no_grad():
            features = self.extractor(batch_tensor)
        
        # 将张量转换为numpy数组
        features_np = features.cpu().numpy()
        
        return features_np
    
    def process_directory(self, input_dir, output_file, file_format='pickle'):
        """
        处理整个目录中的图像，并将特征向量保存到文件
        
        参数:
            input_dir: 包含分割后的骑行者图像的目录
            output_file: 要保存特征向量的输出文件路径
            file_format: 输出文件格式 ('pickle', 'h5', 'json')
        """
        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        
        image_paths = [str(f) for f in image_files]
        
        if len(image_paths) == 0:
            print(f"警告: 在 {input_dir} 未找到图像文件")
            return
        
        print(f"开始处理 {len(image_paths)} 张图像...")
        
        # 存储结果
        results = {}
        
        # 分批处理图像
        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_filenames = [os.path.basename(p) for p in batch_paths]
            
            # 提取特征
            features = self.extract_batch(batch_paths)
            
            # 存储结果
            for j, filename in enumerate(batch_filenames):
                results[filename] = features[j]
        
        # 保存特征向量到文件
        self._save_features(results, output_file, file_format)
        
        print(f"特征向量已保存到 {output_file}")
        print(f"特征向量维度: {list(results.values())[0].shape}")
    
    def _save_features(self, features_dict, output_file, file_format):
        """
        将特征向量保存到文件
        
        参数:
            features_dict: 文件名到特征向量的字典
            output_file: 输出文件路径
            file_format: 文件格式 ('pickle', 'h5', 'json')
        """
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        if file_format == 'pickle':
            with open(output_file, 'wb') as f:
                pickle.dump(features_dict, f)
        
        elif file_format == 'h5':
            with h5py.File(output_file, 'w') as f:
                for filename, features in features_dict.items():
                    # 使用文件名作为数据集名称，但需要替换特殊字符
                    dataset_name = filename.replace('/', '_').replace('\\', '_')
                    f.create_dataset(dataset_name, data=features)
        
        elif file_format == 'json':
            # 将numpy数组转换为列表
            json_dict = {k: v.tolist() for k, v in features_dict.items()}
            with open(output_file, 'w') as f:
                json.dump(json_dict, f)
        
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")

def main():
    parser = argparse.ArgumentParser(description="从YOLO-seg分割的骑行者图像中提取特征向量")
    
    # 必需参数
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含分割后骑行者图像的目录')
    parser.add_argument('--output_file', type=str, required=True,
                        help='保存特征向量的输出文件路径')
    
    # 可选参数
    parser.add_argument('--model', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='用于特征提取的CNN模型')
    parser.add_argument('--format', type=str, default='pickle',
                        choices=['pickle', 'h5', 'json'],
                        help='输出文件格式')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'], 
                        help='运行设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 初始化特征提取器
    extractor = YoloSegFeatureExtractor(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 处理目录
    extractor.process_directory(
        input_dir=args.input_dir,
        output_file=args.output_file,
        file_format=args.format
    )

if __name__ == '__main__':
    main() 