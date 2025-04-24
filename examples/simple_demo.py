"""
骑行者颜色提取系统简单演示脚本
"""

import os
import sys
import json
import argparse
from PIL import Image

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.color_extraction.segmentation_model import CyclistSegmentationModel
from src.color_extraction.color_extraction_model import ColorExtractionModel
import config

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="骑行者颜色提取演示")
    
    parser.add_argument("--image", type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--model", type=str, 
                        default=os.path.join(config.MODEL_DIR, "segmentation_model_best.pth"),
                        help="分割模型路径")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 (cuda 或 cpu)")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 确保输入图像存在
    if not os.path.exists(args.image):
        print(f"错误：输入图像 '{args.image}' 不存在")
        return
    
    print("初始化模型...")
    segmentation_model = None
    
    # 如果模型文件存在，加载分割模型
    if os.path.exists(args.model):
        segmentation_model = CyclistSegmentationModel(args.model, device=args.device)
        print(f"成功加载分割模型：{args.model}")
    else:
        print(f"警告：分割模型 '{args.model}' 不存在，将使用启发式分割")
    
    # 初始化颜色提取模型
    extraction_model = ColorExtractionModel(segmentation_model=segmentation_model)
    
    print(f"处理图像：{args.image}")
    
    # 提取颜色
    results = extraction_model.extract_colors(args.image)
    
    # 打印结果
    print("\n颜色识别结果:")
    print("-" * 30)
    
    for part, colors in results.items():
        if colors:
            color_name, confidence = colors[0]
            print(f"{part.upper()}: {color_name} (置信度: {confidence:.2f})")
        else:
            print(f"{part.upper()}: 未检测到颜色")
    
    print("\n原始结果:")
    print(json.dumps(
        {part: [(color, float(conf)) for color, conf in colors] for part, colors in results.items()},
        indent=2,
        ensure_ascii=False
    ))

if __name__ == "__main__":
    main() 