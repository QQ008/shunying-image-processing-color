"""
骑行者颜色提取系统API接口
"""

import os
import sys
import time
import json
import logging
import base64
import io
import tempfile
import uvicorn
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import torch
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import setup_logger, log_color_result, create_batch_id
from utils.color_utils import visualize_colors
from src.color_extraction.segmentation_model import CyclistSegmentationModel
from src.color_extraction.color_extraction_model import ColorExtractionModel
import config

# 创建FastAPI应用
app = FastAPI(
    title="骑行者颜色提取API",
    description="提取骑行者图像中的头盔、上衣和裤子颜色",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class ColorExtractionRequest(BaseModel):
    image_base64: str
    visualize: Optional[bool] = False

# 定义响应模型
class ColorResult(BaseModel):
    color: str
    confidence: float

class ColorExtractionResponse(BaseModel):
    helmet: ColorResult
    upper_body: ColorResult
    lower_body: ColorResult
    processing_time: float
    visualization_base64: Optional[str] = None

# 全局变量
extraction_model = None
logger = None

def init_models():
    """
    初始化模型
    """
    global extraction_model, logger
    
    # 设置日志
    log_dir = os.path.dirname(config.LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("api", config.LOG_FILE, level=getattr(logging, config.LOG_LEVEL))
    
    # 检查CUDA可用性
    device = torch.device(config.DEVICE)
    logger.info(f"使用设备: {device}")
    
    # 加载分割模型
    model_path = os.path.join(config.MODEL_DIR, "segmentation_model_best.pth")
    segmentation_model = None
    
    if os.path.exists(model_path):
        logger.info(f"加载分割模型: {model_path}")
        segmentation_model = CyclistSegmentationModel(model_path=model_path, device=device)
    else:
        logger.warning(f"分割模型文件不存在: {model_path}，将使用启发式分割")
    
    # 初始化颜色提取模型
    logger.info("初始化颜色提取模型...")
    extraction_model = ColorExtractionModel(segmentation_model=segmentation_model, device=device)
    
    logger.info("模型初始化完成")

@app.on_event("startup")
async def startup_event():
    """
    应用启动时初始化模型
    """
    init_models()

@app.get("/")
async def root():
    """
    API根路径，返回欢迎信息
    """
    return {"message": "欢迎使用骑行者颜色提取API"}

@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/extract_colors", response_model=ColorExtractionResponse)
async def extract_colors(request: ColorExtractionRequest):
    """
    从Base64编码的图像中提取颜色
    """
    global extraction_model, logger
    
    # 确保模型已初始化
    if extraction_model is None:
        init_models()
    
    start_time = time.time()
    
    try:
        # 解码Base64图像
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
        
        # 提取颜色
        result = extraction_model.extract_colors(temp_path)
        
        # 记录结果
        if logger:
            log_color_result(logger, temp_path, result)
        
        # 准备响应
        response = {}
        visualization_base64 = None
        
        # 为每个部位获取主要颜色
        for part, colors in result.items():
            if colors:
                color_name, confidence = colors[0]
                response[part] = ColorResult(color=color_name, confidence=float(confidence))
            else:
                response[part] = ColorResult(color="unknown", confidence=0.0)
        
        # 生成可视化（如果请求）
        if request.visualize:
            # 读取临时图像
            original_image = cv2.imread(temp_path)
            
            # 创建可视化图像
            vis_image = np.zeros((300, 600, 3), dtype=np.uint8)
            
            # 在左侧绘制原始图像
            h, w = original_image.shape[:2]
            aspect_ratio = h / w
            vis_width = 280
            vis_height = int(vis_width * aspect_ratio)
            resized_image = cv2.resize(original_image, (vis_width, vis_height))
            
            # 计算图像位置（居中）
            y_offset = (300 - vis_height) // 2
            vis_image[y_offset:y_offset+vis_height, 10:10+vis_width] = resized_image
            
            # 在右侧创建颜色区域
            x_offset = 300
            y_pos = 50
            
            # 为每个部位绘制颜色块和标签
            for part, color_result in response.items():
                # 绘制标签
                label = f"{part}: {color_result.color} ({color_result.confidence:.2f})"
                cv2.putText(vis_image, label, (x_offset, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 获取颜色RGB值并转换为BGR
                from utils.color_utils import color_name_to_rgb
                rgb = color_name_to_rgb(color_result.color)
                bgr = (rgb[2], rgb[1], rgb[0])
                
                # 绘制颜色块
                cv2.rectangle(vis_image, (x_offset, y_pos+10), (x_offset+100, y_pos+40), bgr, -1)
                
                y_pos += 80
            
            # 编码为Base64
            _, buffer = cv2.imencode('.png', vis_image)
            visualization_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 删除临时文件
        os.unlink(temp_path)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建完整响应
        full_response = ColorExtractionResponse(
            helmet=response["helmet"],
            upper_body=response["upper_body"],
            lower_body=response["lower_body"],
            processing_time=processing_time,
            visualization_base64=visualization_base64
        )
        
        return full_response
    
    except Exception as e:
        if logger:
            logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_colors_file")
async def extract_colors_file(
    file: UploadFile = File(...),
    visualize: bool = Form(False)
):
    """
    从上传的图像文件中提取颜色
    """
    global extraction_model, logger
    
    # 确保模型已初始化
    if extraction_model is None:
        init_models()
    
    start_time = time.time()
    
    try:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # 提取颜色
        result = extraction_model.extract_colors(temp_path)
        
        # 记录结果
        if logger:
            log_color_result(logger, temp_path, result)
        
        # 准备响应
        response = {}
        visualization_base64 = None
        
        # 为每个部位获取主要颜色
        for part, colors in result.items():
            if colors:
                color_name, confidence = colors[0]
                response[part] = {"color": color_name, "confidence": float(confidence)}
            else:
                response[part] = {"color": "unknown", "confidence": 0.0}
        
        # 与Base64接口不同，文件版本返回所有检测到的颜色
        full_result = {}
        for part, colors in result.items():
            full_result[part] = [
                {"color": color, "confidence": float(conf)} 
                for color, conf in colors
            ]
        
        # 生成可视化（如果请求）
        if visualize:
            # 读取临时图像
            original_image = cv2.imread(temp_path)
            
            # 为每个部位创建颜色条
            part_visualizations = {}
            for part, colors in result.items():
                if not colors:
                    continue
                
                # 提取部位的BGR颜色值和置信度
                part_colors = []
                percentages = []
                
                for color_name, confidence in colors:
                    # 获取颜色的BGR值
                    from utils.color_utils import color_name_to_rgb
                    rgb = color_name_to_rgb(color_name)
                    bgr = (rgb[2], rgb[1], rgb[0])  # 转换为BGR
                    part_colors.append(bgr)
                    percentages.append(confidence)
                
                # 生成颜色可视化
                color_vis = visualize_colors(part_colors, percentages)
                
                # 编码为Base64
                _, buffer = cv2.imencode('.png', color_vis)
                part_visualizations[part] = base64.b64encode(buffer).decode('utf-8')
            
            # 添加到响应
            response["visualizations"] = part_visualizations
        
        # 删除临时文件
        os.unlink(temp_path)
        
        # 添加详细结果和处理时间
        response["full_result"] = full_result
        response["processing_time"] = time.time() - start_time
        
        return response
    
    except Exception as e:
        if logger:
            logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """
    启动API服务器
    """
    # 从环境变量获取主机和端口，如果没有则使用默认值
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    uvicorn.run("api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start() 