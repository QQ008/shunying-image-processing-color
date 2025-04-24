"""
颜色处理工具函数集
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from collections import Counter
import colorsys

# 预定义的颜色映射表
COLOR_MAP = {
    "black": ([0, 0, 0], [50, 50, 50]),
    "white": ([200, 200, 200], [255, 255, 255]),
    "red": ([0, 0, 150], [80, 80, 255]),
    "green": ([0, 150, 0], [80, 255, 80]),
    "blue": ([150, 0, 0], [255, 80, 80]),
    "yellow": ([0, 150, 150], [80, 255, 255]),
    "orange": ([0, 80, 200], [80, 150, 255]),
    "purple": ([130, 0, 75], [255, 50, 150]),
    "pink": ([150, 50, 150], [255, 120, 255]),
    "brown": ([0, 50, 100], [80, 120, 150]),
    "gray": ([80, 80, 80], [150, 150, 150]),
    "cyan": ([150, 150, 0], [255, 255, 80])
}

def extract_dominant_colors(image, k=5, mask=None):
    """
    提取图像中的主要颜色
    
    Args:
        image: 输入图像 (numpy数组, BGR格式)
        k: 提取颜色数量
        mask: 掩码图像 (可选)
        
    Returns:
        colors: 主要颜色列表 (BGR格式)
        percentages: 每种颜色的百分比
    """
    # 确保图像是3通道
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是3通道BGR格式")
    
    # 转换为RGB以便处理
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 重塑图像为二维数组
    if mask is not None:
        pixels = image_rgb[mask > 0].reshape(-1, 3)
    else:
        pixels = image_rgb.reshape(-1, 3)
    
    # 防止空图像
    if len(pixels) == 0:
        return [], []
    
    # 使用K-means聚类提取主要颜色
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    
    # 获取颜色并计算百分比
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    count = Counter(labels)
    
    # 计算每种颜色的百分比
    total_pixel_count = sum(count.values())
    percentages = [count[i] / total_pixel_count for i in range(k)]
    
    # 按百分比排序
    colors_with_percentages = sorted(
        zip(colors, percentages), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 分离颜色和百分比
    colors = [c for c, _ in colors_with_percentages]
    percentages = [p for _, p in colors_with_percentages]
    
    # 转回BGR格式
    colors_bgr = [color[::-1] for color in colors]
    
    return colors_bgr, percentages

def classify_color(color_bgr):
    """
    将BGR颜色分类为预定义的颜色类别
    
    Args:
        color_bgr: BGR格式的颜色
        
    Returns:
        color_name: 颜色名称
        confidence: 匹配置信度
    """
    color_rgb = color_bgr[::-1]  # 转换为RGB
    
    best_match = None
    best_distance = float('inf')
    
    # 计算与预定义颜色的距离
    for color_name, (lower, upper) in COLOR_MAP.items():
        # 检查颜色是否在范围内
        lower = np.array(lower)
        upper = np.array(upper)
        if np.all(color_rgb >= lower) and np.all(color_rgb <= upper):
            # 计算到范围中心的距离
            center = (np.array(lower) + np.array(upper)) / 2
            distance = np.linalg.norm(color_rgb - center)
            
            if distance < best_distance:
                best_distance = distance
                best_match = color_name
    
    # 如果没有直接匹配，查找最接近的颜色
    if best_match is None:
        for color_name, (lower, upper) in COLOR_MAP.items():
            center = (np.array(lower) + np.array(upper)) / 2
            distance = np.linalg.norm(color_rgb - center)
            
            if distance < best_distance:
                best_distance = distance
                best_match = color_name
    
    # 计算置信度分数
    max_possible_distance = 441.67  # 3D RGB空间中的最大距离 (255*sqrt(3))
    confidence = 1 - (best_distance / max_possible_distance)
    
    return best_match, confidence

def color_name_to_rgb(color_name):
    """
    将颜色名称转换为RGB值
    
    Args:
        color_name: 颜色名称
        
    Returns:
        rgb: RGB颜色值
    """
    try:
        # 尝试从webcolors获取颜色
        rgb = webcolors.name_to_rgb(color_name)
        return rgb
    except ValueError:
        # 如果不是标准颜色名称，使用我们的映射
        if color_name in COLOR_MAP:
            lower, upper = COLOR_MAP[color_name]
            # 返回范围的中间值
            return tuple((np.array(lower) + np.array(upper)) // 2)
    
    # 默认返回灰色
    return (128, 128, 128)

def visualize_colors(colors, percentages, width=300, height=100):
    """
    生成颜色可视化图像
    
    Args:
        colors: BGR颜色列表
        percentages: 每种颜色的百分比
        width: 输出图像宽度
        height: 输出图像高度
        
    Returns:
        visualization: 可视化图像
    """
    # 创建空白图像
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 如果没有颜色，返回空白图像
    if not colors:
        return visualization
    
    # 计算每种颜色的宽度
    cum_percent = 0
    for i, (color, percent) in enumerate(zip(colors, percentages)):
        # 计算当前颜色块的起始和结束位置
        start_x = int(cum_percent * width)
        cum_percent += percent
        end_x = int(cum_percent * width)
        
        # 填充矩形
        visualization[:, start_x:end_x] = color
    
    return visualization 