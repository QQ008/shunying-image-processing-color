"""
骑行者颜色提取系统推理接口
"""

import os
import sys
import time
import json
import argparse
import logging
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import setup_logger, log_processing_stats, log_color_result, create_batch_id
from utils.color_utils import visualize_colors
from src.color_extraction.segmentation_model import CyclistSegmentationModel
from src.color_extraction.color_extraction_model import ColorExtractionModel
from src.color_extraction.data_processor import preprocess_image
import config

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="骑行者颜色提取系统")
    
    # 输入输出参数
    parser.add_argument("--input", type=str, default=config.INPUT_DIR,
                        help="输入图像目录或单张图像路径")
    parser.add_argument("--output", type=str, default=config.OUTPUT_DIR,
                        help="输出结果目录")
    
    # 模型参数
    parser.add_argument("--segmentation_model", type=str, 
                        default=os.path.join(config.MODEL_DIR, "segmentation_model_best.pth"),
                        help="分割模型路径")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="计算设备")
    
    # 处理参数
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="批处理大小")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS,
                        help="并行处理的工作线程数")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成颜色可视化图像")
    
    return parser.parse_args()

def process_single_image(image_path, extraction_model, output_dir, visualize=False, logger=None):
    """
    处理单张图像
    
    Args:
        image_path: 图像路径
        extraction_model: 颜色提取模型
        output_dir: 输出目录
        visualize: 是否生成可视化
        logger: 日志记录器
        
    Returns:
        result: 处理结果
        success: 是否成功
    """
    try:
        # 提取颜色
        result = extraction_model.extract_colors(image_path)
        
        # 记录结果
        if logger:
            log_color_result(logger, image_path, result)
        
        # 保存结果
        if output_dir:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 保存JSON结果
            result_json = {}
            for part, colors in result.items():
                if colors:
                    result_json[part] = {
                        "color": colors[0][0],
                        "confidence": float(colors[0][1])
                    }
                else:
                    result_json[part] = {
                        "color": "unknown",
                        "confidence": 0.0
                    }
            
            json_path = os.path.join(output_dir, f"{base_name}_colors.json")
            with open(json_path, 'w') as f:
                json.dump(result_json, f, indent=4)
            
            # 生成可视化图像
            if visualize:
                # 读取原始图像
                image = cv2.imread(image_path)
                if image is not None:
                    # 为每个部位创建颜色条
                    for part, colors in result.items():
                        if not colors:
                            continue
                        
                        # 提取部位的BGR颜色值和置信度
                        part_colors = []
                        percentages = []
                        
                        for color_name, confidence in colors:
                            # 获取颜色的BGR值（这里需要实现从颜色名称到BGR的映射）
                            from utils.color_utils import color_name_to_rgb
                            rgb = color_name_to_rgb(color_name)
                            bgr = (rgb[2], rgb[1], rgb[0])  # 转换为BGR
                            part_colors.append(bgr)
                            percentages.append(confidence)
                        
                        # 生成颜色可视化
                        color_vis = visualize_colors(part_colors, percentages)
                        
                        # 保存可视化图像
                        vis_path = os.path.join(output_dir, f"{base_name}_{part}_colors.png")
                        cv2.imwrite(vis_path, color_vis)
        
        return result, True
    
    except Exception as e:
        if logger:
            logger.error(f"处理图像失败 [{os.path.basename(image_path)}]: {str(e)}")
        return None, False

def process_batch(image_paths, extraction_model, output_dir, visualize=False, workers=4, logger=None):
    """
    批量处理图像
    
    Args:
        image_paths: 图像路径列表
        extraction_model: 颜色提取模型
        output_dir: 输出目录
        visualize: 是否生成可视化
        workers: 工作线程数
        logger: 日志记录器
        
    Returns:
        results: 处理结果列表
        stats: 处理统计信息
    """
    results = []
    success_count = 0
    error_count = 0
    
    start_time = time.time()
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_path = {
            executor.submit(
                process_single_image, 
                path, 
                extraction_model, 
                output_dir, 
                visualize, 
                logger
            ): path for path in image_paths
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="处理图像"):
            path = future_to_path[future]
            try:
                result, success = future.result()
                if success:
                    results.append((path, result))
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                if logger:
                    logger.error(f"处理任务失败 [{os.path.basename(path)}]: {str(e)}")
                error_count += 1
    
    # 记录处理统计信息
    if logger:
        log_processing_stats(
            logger, 
            start_time, 
            len(image_paths), 
            success_count, 
            error_count
        )
    
    stats = {
        "total": len(image_paths),
        "success": success_count,
        "error": error_count,
        "time": time.time() - start_time
    }
    
    return results, stats

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.dirname(config.LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志记录器
    batch_id = create_batch_id()
    log_file = os.path.join(log_dir, f"inference_{batch_id}.log")
    logger = setup_logger("inference", log_file, level=getattr(logging, config.LOG_LEVEL))
    
    # 记录开始处理
    logger.info(f"开始处理批次 {batch_id}")
    logger.info(f"参数: {vars(args)}")
    
    # 初始化模型
    device = torch.device(args.device)
    
    # 加载分割模型
    logger.info("加载分割模型...")
    segmentation_model = CyclistSegmentationModel(
        model_path=args.segmentation_model,
        device=device
    )
    
    # 初始化颜色提取模型
    logger.info("初始化颜色提取模型...")
    extraction_model = ColorExtractionModel(
        segmentation_model=segmentation_model,
        device=device
    )
    
    # 获取输入图像路径
    if os.path.isdir(args.input):
        # 如果输入是目录，获取所有图像文件
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_paths.extend(glob.glob(os.path.join(args.input, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(args.input, f"*.{ext.upper()}")))
    else:
        # 如果输入是单个文件
        image_paths = [args.input]
    
    logger.info(f"找到 {len(image_paths)} 个图像文件")
    
    # 处理图像
    logger.info("开始处理图像...")
    results, stats = process_batch(
        image_paths,
        extraction_model,
        args.output,
        args.visualize,
        args.workers,
        logger
    )
    
    # 保存汇总结果
    summary_path = os.path.join(args.output, f"summary_{batch_id}.json")
    summary = {
        "batch_id": batch_id,
        "stats": stats,
        "results": [
            {
                "image": os.path.basename(path),
                "colors": {
                    part: {
                        "color": colors[0][0] if colors else "unknown",
                        "confidence": float(colors[0][1]) if colors else 0.0
                    } for part, colors in result.items()
                }
            } for path, result in results
        ]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"处理完成，结果保存至 {args.output}")
    logger.info(f"处理统计: 总数={stats['total']}, 成功={stats['success']}, "
               f"失败={stats['error']}, 耗时={stats['time']:.2f}秒")

if __name__ == "__main__":
    main() 