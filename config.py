"""
颜色识别系统的配置文件
"""

# 输入输出路径配置
INPUT_DIR = "data/input"          # YOLO-SEG裁剪后的图片输入目录
OUTPUT_DIR = "data/output"        # 颜色分析结果输出目录
MODEL_DIR = "models"              # 模型存储目录

# 模型配置
COLOR_MODEL_TYPE = "segmentation"   # segmentation 或 classification
BATCH_SIZE = 32                     # 批处理大小，可根据GPU内存调整
NUM_WORKERS = 4                     # 数据加载线程数
DEVICE = "cuda"                     # 使用 "cuda" 或 "cpu"

# 颜色分析配置
COLOR_CLASSES = [
    # 基本颜色类别
    "black", "white", "red", "green", "blue", "yellow", 
    "orange", "purple", "pink", "brown", "gray", "cyan"
]

# 身体部位分割配置
BODY_PARTS = {
    "helmet": 0,     # 头盔区域
    "upper_body": 1, # 上衣区域
    "lower_body": 2  # 裤子区域
}

# 颜色识别阈值
COLOR_THRESHOLD = 0.2      # 颜色检测的置信度阈值
MIN_AREA_RATIO = 0.05      # 最小区域比例，过滤噪点

# 系统性能配置
MAX_IMAGES_PER_DAY = 50000   # 每日最大处理图片数
PROCESSING_TIMEOUT = 30      # 单张图片处理超时时间(秒)

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = "logs/color_extraction.log" 