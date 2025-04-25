import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, Optional, List, Union

# ----- 阶段 1 & 2: 特征提取（占位符） -----

class FeatureExtractor(nn.Module):
    """
    基于预训练CNN的特征提取器。
    从裁剪后的骑手图像中提取固定长度的特征向量。
    """
    def __init__(self, 
                 model_name: str = 'mobilenet_v3_small', 
                 pretrained: bool = True, 
                 freeze_backbone: bool = True):
        """
        参数：
            model_name: 骨干网络的名称 ('mobilenet_v3_small', 'mobilenet_v3_large', 等)
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结骨干网络的权重
        """
        super().__init__()
        
        # 加载预训练模型
        if model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.feature_dim = 576  # mobilenet_v3_small的特征维度
        elif model_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            self.feature_dim = 960  # mobilenet_v3_large的特征维度
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 移除分类器以获取特征
        self.backbone.classifier = nn.Identity()
        
        # 如果指定则冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 定义预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        从输入图像中提取特征
        
        参数：
            x: 形状为 [batch_size, 3, H, W] 的输入张量
            
        返回：
            形状为 [batch_size, feature_dim] 的特征向量
        """
        features = self.backbone(x)
        return features
    
    def preprocess(self, image):
        """对PIL图像或张量应用预处理"""
        if not isinstance(image, torch.Tensor):
            return self.transform(image)
        return image

# ----- 阶段 3: 颜色属性预测MLP模型 -----

class ColorPredictorMLP(nn.Module):
    """
    多头MLP模型，用于从输入特征向量预测多个身体部位的颜色属性。

    参数：
        feature_dim (int): 输入特征向量F的维度
        num_classes (int): 颜色类别数量（包括'不清楚/未穿戴'）
        shared_hidden_dim (int, optional): 共享隐藏层的维度。如果为None，则不使用共享层。
        head_hidden_dim (int): 每个预测头中隐藏层的维度
        dropout_rate (float): Dropout率
        part_names (List[str], optional): 要预测颜色的身体部位名称列表
    """
    def __init__(self,
                 feature_dim: int,
                 num_classes: int,
                 shared_hidden_dim: Optional[int] = None,
                 head_hidden_dim: int = 256,
                 dropout_rate: float = 0.4,
                 part_names: List[str] = None):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.shared_hidden_dim = shared_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        
        # 如果未指定则使用默认部位名称
        self.part_names = part_names or ['helmet', 'jersey', 'shorts', 'shoes']

        # --- 可选的共享层 ---
        if shared_hidden_dim:
            self.shared_layers = nn.Sequential(
                nn.Linear(feature_dim, shared_hidden_dim),
                nn.BatchNorm1d(shared_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            head_input_dim = shared_hidden_dim
        else:
            self.shared_layers = None
            head_input_dim = feature_dim  # 如果没有共享层，头部直接接收原始特征

        # --- 独立的预测头 ---
        self.prediction_heads = nn.ModuleDict({
            part: self._create_prediction_head(
                head_input_dim, head_hidden_dim, num_classes, dropout_rate
            ) for part in self.part_names
        })

    def _create_prediction_head(self, input_dim, hidden_dim, output_dim, dropout_rate):
        """创建单个预测头的辅助函数"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)  # 输出层，输出logits
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        模型的前向传播。

        参数：
            x (torch.Tensor): 输入特征向量F，形状为 [batch_size, feature_dim]

        返回：
            Dict[str, torch.Tensor]: 包含每个身体部位logits的字典。
                                     每个张量的形状为 [batch_size, num_classes]。
        """
        # 通过共享层（如果存在）
        if self.shared_layers:
            shared_features = self.shared_layers(x)
            head_input = shared_features
        else:
            head_input = x

        # 通过独立的预测头
        logits = {part: self.prediction_heads[part](head_input) for part in self.part_names}
        
        return logits


# ----- 完整模型: 特征提取 + 颜色预测 -----

class RiderColorPredictor(nn.Module):
    """
    用于从图像预测骑手颜色的端到端模型。
    结合特征提取和颜色预测功能。
    """
    def __init__(self,
                 feature_extractor_name: str = 'mobilenet_v3_small',
                 num_classes: int = 14,
                 shared_hidden_dim: Optional[int] = 256,
                 head_hidden_dim: int = 128,
                 dropout_rate: float = 0.4,
                 freeze_backbone: bool = True,
                 part_names: List[str] = None):
        super().__init__()
        
        # 特征提取器组件
        self.feature_extractor = FeatureExtractor(
            model_name=feature_extractor_name,
            pretrained=True,
            freeze_backbone=freeze_backbone
        )
        
        # 用于颜色预测的MLP
        self.color_predictor = ColorPredictorMLP(
            feature_dim=self.feature_extractor.feature_dim,
            num_classes=num_classes,
            shared_hidden_dim=shared_hidden_dim,
            head_hidden_dim=head_hidden_dim,
            dropout_rate=dropout_rate,
            part_names=part_names
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        端到端模型的前向传播。
        
        参数：
            x: 形状为 [batch_size, 3, H, W] 的输入图像
            
        返回：
            每个身体部位的logits字典
        """
        features = self.feature_extractor(x)
        logits = self.color_predictor(features)
        return logits


# ----- 使用示例 -----
if __name__ == '__main__':
    # --- 模型参数 ---
    NUM_CLASSES = 14       # 颜色类别数量
    SHARED_HIDDEN = 256    # 共享隐藏层的维度
    HEAD_HIDDEN = 128      # 预测头隐藏层的维度
    DROPOUT = 0.4          # Dropout率
    BATCH_SIZE = 4         # 批量大小
    
    # --- 创建完整模型 ---
    model = RiderColorPredictor(
        feature_extractor_name='mobilenet_v3_small',
        num_classes=NUM_CLASSES,
        shared_hidden_dim=SHARED_HIDDEN,
        head_hidden_dim=HEAD_HIDDEN,
        dropout_rate=DROPOUT,
        freeze_backbone=True
    )
    
    # 打印模型结构
    print(f"模型初始化，特征维度: {model.feature_extractor.feature_dim}")
    
    # --- 创建虚拟输入数据 ---
    # 形状: [batch_size, channels, height, width]
    dummy_images = torch.randn(BATCH_SIZE, 3, 224, 224)
    
    # --- 模型推理 ---
    model.eval()  # 设置为评估模式（禁用dropout等）
    with torch.no_grad():  # 不计算梯度
        output_logits = model(dummy_images)
    
    # --- 打印输出形状 ---
    print("\n模型输出logits:")
    for part, logits in output_logits.items():
        print(f"  - {part}: 形状 {logits.shape}")  # 应该是 [BATCH_SIZE, NUM_CLASSES]
    
    # --- 如何获取预测概率和类别 ---
    # 对logits应用Softmax以获取概率
    probabilities = {part: torch.softmax(logits, dim=1) for part, logits in output_logits.items()}
    # 获取最高概率类别的索引
    predicted_classes = {part: torch.argmax(probs, dim=1) for part, probs in probabilities.items()}
    
    print("\n预测概率（第一个样本）:")
    for part, probs in probabilities.items():
        print(f"  - {part}: {probs[0].numpy().round(3)}")
    
    print("\n预测类别索引（所有样本）:")
    for part, classes in predicted_classes.items():
        print(f"  - {part}: {classes.numpy()}") 