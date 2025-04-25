import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import json
from typing import Dict, List, Tuple, Union, Optional

from color_predictor import RiderColorPredictor

class ColorPredictor:
    """
    用于对骑手图像进行颜色预测的类。
    
    该类加载一个训练好的模型，并提供方法来预测
    骑手图像中不同身体部位的颜色。
    """
    def __init__(self, 
                 model_path: str, 
                 device: str = None,
                 color_map_path: str = None,
                 confidence_threshold: float = 0.5):
        """
        初始化颜色预测器。
        
        参数：
            model_path: 保存的模型检查点路径
            device: 运行推理的设备 ('cuda' 或 'cpu')
            color_map_path: 将类别索引映射到颜色名称的JSON文件路径
            confidence_threshold: 置信预测的阈值
        """
        # 设置设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 如果提供了颜色映射，则加载
        self.color_map = None
        if color_map_path and os.path.exists(color_map_path):
            with open(color_map_path, 'r') as f:
                self.color_map = json.load(f)
            print(f"加载了包含 {len(self.color_map)} 个颜色类别的颜色映射")
        
        # 设置置信度阈值
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 从模型获取部位名称
        self.part_names = self.model.color_predictor.part_names
        print(f"模型预测以下部位的颜色: {', '.join(self.part_names)}")
        
        # 设置图像转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """
        从检查点加载模型。
        
        参数：
            model_path: 模型检查点路径
        
        返回：
            加载的模型
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从检查点确定模型架构
        # 这假设检查点包含关于模型架构的信息
        # 在实际实现中，您可能需要显式地保存和加载这些信息
        
        # 现在，让我们假设标准配置
        model = RiderColorPredictor(
            feature_extractor_name='mobilenet_v3_small',
            num_classes=14,  # 这应该与您的训练配置匹配
        )
        
        # 从检查点加载状态字典
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"从 {model_path} 加载模型（训练了 {checkpoint['epoch']} 个周期）")
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理图像以进行推理。
        
        参数：
            image_path: 图像文件路径
        
        返回：
            预处理后的图像张量
        """
        # 加载并转换为RGB
        image = Image.open(image_path).convert('RGB')
        
        # 应用转换
        image_tensor = self.transform(image)
        
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def predict(self, image_path: str) -> Dict[str, Dict[str, Union[int, float, str]]]:
        """
        预测单个图像的颜色。
        
        参数：
            image_path: 图像文件路径
        
        返回：
            包含每个身体部位预测的字典
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 处理输出
        predictions = {}
        for part, logits in outputs.items():
            # 应用softmax获取概率
            probs = torch.softmax(logits, dim=1)
            
            # 获取预测类别和置信度
            confidence, class_idx = torch.max(probs, dim=1)
            
            # 转换为Python类型
            class_idx = class_idx.item()
            confidence = confidence.item()
            
            # 如果有颜色映射，则获取颜色名称
            color_name = "unknown"
            if self.color_map and str(class_idx) in self.color_map:
                color_name = self.color_map[str(class_idx)]
            
            # 存储预测
            predictions[part] = {
                'class_idx': class_idx,
                'confidence': confidence,
                'color_name': color_name,
                'is_confident': confidence >= self.confidence_threshold
            }
            
            # 添加所有概率以进行详细分析
            predictions[part]['all_probs'] = probs[0].cpu().numpy().tolist()
        
        return predictions
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Dict[str, Union[int, float, str]]]]:
        """
        预测多个图像的颜色。
        
        参数：
            image_paths: 图像文件路径列表
        
        返回：
            预测字典列表，每个图像一个
        """
        results = []
        for image_path in image_paths:
            predictions = self.predict(image_path)
            results.append({
                'image_path': image_path,
                'predictions': predictions
            })
        return results
    
    def print_predictions(self, predictions: Dict[str, Dict[str, Union[int, float, str]]]):
        """
        以人类可读的格式打印预测结果。
        
        参数：
            predictions: 来自predict()的预测字典
        """
        print("颜色预测:")
        print("=================")
        for part, pred in predictions.items():
            confidence_str = f"{pred['confidence']:.2f}"
            status = "✓" if pred['is_confident'] else "?"
            
            if pred['is_confident']:
                print(f"{part:10s}: {pred['color_name']:10s} ({confidence_str}) {status}")
            else:
                print(f"{part:10s}: {"uncertain":10s} ({confidence_str}) {status}")
        print("=================")


# ----- 命令行界面 -----

def main():
    parser = argparse.ArgumentParser(description="使用骑手颜色预测器运行推理")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='保存的模型检查点路径')
    parser.add_argument('--image_path', type=str, required=True,
                        help='要预测的图像或图像目录的路径')
    
    # 可选参数
    parser.add_argument('--color_map', type=str, default=None,
                        help='将类别索引映射到颜色名称的JSON文件路径')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='置信预测的阈值')
    parser.add_argument('--output_json', type=str, default=None,
                        help='保存预测结果为JSON的路径')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'], help='运行推理的设备')
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = ColorPredictor(
        model_path=args.model_path,
        device=args.device,
        color_map_path=args.color_map,
        confidence_threshold=args.confidence_threshold
    )
    
    # 收集图像路径
    image_paths = []
    if os.path.isdir(args.image_path):
        # 如果提供了目录，则处理其中的所有图像
        for filename in os.listdir(args.image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.image_path, filename))
    else:
        # 如果提供了单个图像
        image_paths.append(args.image_path)
    
    # 运行预测
    results = predictor.batch_predict(image_paths)
    
    # 打印预测
    for result in results:
        print(f"\n图像: {result['image_path']}")
        predictor.print_predictions(result['predictions'])
    
    # 如果需要，将结果保存为JSON
    if args.output_json:
        # 将numpy数组转换为列表以进行JSON序列化
        for result in results:
            for part, pred in result['predictions'].items():
                if isinstance(pred['all_probs'], np.ndarray):
                    pred['all_probs'] = pred['all_probs'].tolist()
        
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n已将预测结果保存到 {args.output_json}")

if __name__ == '__main__':
    main() 