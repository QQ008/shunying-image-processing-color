"""
日志工具模块
"""

import logging
import os
import sys
import time
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """
    设置日志工具
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 设置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 配置日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_processing_stats(logger, start_time, image_count, success_count, error_count):
    """
    记录处理统计信息
    
    Args:
        logger: 日志记录器
        start_time: 开始时间
        image_count: 处理的图像总数
        success_count: 成功处理的图像数
        error_count: 处理失败的图像数
    """
    elapsed_time = time.time() - start_time
    images_per_second = image_count / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"处理完成: 总数={image_count}, 成功={success_count}, 失败={error_count}")
    logger.info(f"处理速度: {images_per_second:.2f} 图像/秒, 总耗时: {elapsed_time:.2f} 秒")
    
    # 预估每日处理能力
    estimated_daily_capacity = int(images_per_second * 3600 * 24)
    logger.info(f"预估每日处理能力: {estimated_daily_capacity} 图像/天")

def log_color_result(logger, image_path, color_results):
    """
    记录颜色识别结果
    
    Args:
        logger: 日志记录器
        image_path: 图像路径
        color_results: 颜色识别结果
    """
    logger.debug(f"图像颜色分析结果 [{os.path.basename(image_path)}]:")
    for part, colors in color_results.items():
        if colors:
            main_color, confidence = colors[0]
            logger.debug(f"  - {part}: {main_color} (置信度: {confidence:.2f})")
        else:
            logger.debug(f"  - {part}: 未检测到颜色")

def log_error(logger, image_path, error_msg):
    """
    记录错误信息
    
    Args:
        logger: 日志记录器
        image_path: 图像路径
        error_msg: 错误信息
    """
    logger.error(f"处理失败 [{os.path.basename(image_path)}]: {error_msg}")

def create_batch_id():
    """
    创建唯一的批处理ID
    
    Returns:
        batch_id: 批处理ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"batch_{timestamp}" 