"""
分割模型：用于将人物图像分割为头盔、上衣和裤子区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class UNet(nn.Module):
    """
    UNet分割模型
    """
    def __init__(self, n_channels=3, n_classes=4, bilinear=True):
        """
        初始化UNet模型
        
        Args:
            n_channels: 输入通道数
            n_classes: 分类数量（背景 + 头盔 + 上衣 + 裤子）
            bilinear: 是否使用双线性上采样
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器 (下采样)
        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 512)

        # 解码器 (上采样)
        self.up1 = self.up(1024, 256, bilinear)
        self.up2 = self.up(512, 128, bilinear)
        self.up3 = self.up(256, 64, bilinear)
        self.up4 = self.up(128, 64, bilinear)
        
        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            x: 输出分割掩码
        """
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        return x

    def double_conv(self, in_channels, out_channels):
        """
        双卷积块
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        """
        下采样块
        """
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels, bilinear=True):
        """
        上采样块
        """
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        return UpBlock(in_channels, out_channels, up)


class UpBlock(nn.Module):
    """
    上采样块
    """
    def __init__(self, in_channels, out_channels, up):
        super(UpBlock, self).__init__()
        self.up = up
        self.conv = UNet.double_conv(self, in_channels, out_channels)

    def forward(self, x1, x2):
        """
        前向传播
        
        Args:
            x1: 上采样输入
            x2: 跳跃连接输入
            
        Returns:
            x: 特征图
        """
        x1 = self.up(x1)
        
        # 调整x1大小以匹配x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CyclistSegmentationModel:
    """
    骑行者分割模型接口
    """
    def __init__(self, model_path=None, device=None):
        """
        初始化分割模型
        
        Args:
            model_path: 模型权重路径
            device: 计算设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        
        # 如果提供了模型路径，加载权重
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型权重: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _create_model(self):
        """
        创建模型
        
        Returns:
            model: UNet模型
        """
        return UNet(n_channels=3, n_classes=4)
    
    def predict(self, image):
        """
        预测分割掩码
        
        Args:
            image: 图像张量 (已预处理)
            
        Returns:
            masks: 分割掩码字典 {部位名称: 掩码}
        """
        with torch.no_grad():
            # 确保输入是张量并放到正确的设备上
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).to(self.device)
            else:
                image = image.to(self.device)
            
            # 前向传播
            output = self.model(image)
            
            # 应用softmax获取概率
            prob = F.softmax(output, dim=1)
            
            # 获取掩码
            masks = self._process_output(prob)
            
            return masks
    
    def _process_output(self, output):
        """
        处理模型输出为分割掩码
        
        Args:
            output: 模型输出
            
        Returns:
            masks: 分割掩码字典 {部位名称: 掩码}
        """
        # 移动到CPU并转换为NumPy数组
        output_np = output.cpu().numpy()
        
        # 每个像素取最大概率的类别
        pred = np.argmax(output_np, axis=1)
        
        # 为每个类别创建掩码
        masks = {}
        for part_name, part_id in config.BODY_PARTS.items():
            # 创建部位掩码（部位ID + 1，因为0是背景）
            mask = (pred == part_id + 1).astype(np.uint8) * 255
            masks[part_name] = mask[0]  # 去掉批次维度
        
        return masks
    
    def segment_image(self, image_path):
        """
        分割图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            original_image: 原始图像
            masks: 分割掩码
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 预处理图像
        processed_image = self._preprocess_image(image)
        
        # 预测分割掩码
        masks = self.predict(processed_image)
        
        return image, masks
    
    def _preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: OpenCV图像
            
        Returns:
            processed_image: 预处理后的图像
        """
        # 调整大小
        resized = cv2.resize(image, (224, 224))
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为RGB
        rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        # 转换为PyTorch期望的格式 (B, C, H, W)
        tensor = np.transpose(rgb, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor

def create_pretrained_segmentation_model():
    """
    创建预训练的分割模型
    
    Returns:
        model: DeepLabV3分割模型
    """
    # 加载预训练的DeepLabV3
    model = models.segmentation.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
    
    # 修改分类器头以适应我们的类别数
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, 4, 1)  # 4类：背景 + 头盔 + 上衣 + 裤子
    
    return model 