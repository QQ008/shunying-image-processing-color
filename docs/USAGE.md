# 骑行者颜色提取系统使用指南

本文档介绍如何使用骑行者颜色提取系统，该系统能够从YOLO-SEG提取的骑行者图像中识别头盔、上衣和裤子的颜色。

## 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐，但不是必须）
- 依赖包（详见requirements.txt）

## 安装步骤

1. 克隆代码库：

```bash
git clone <repository-url>
cd shunying-image-processing-color
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备模型：
   - 如果已有训练好的模型，将模型文件放到 `models/` 目录下
   - 如果需要训练模型，请参考下面的训练部分

## 使用方法

### 1. 图像输入格式

系统期望的输入是由YOLO-SEG模型提取的骑行者图像。这些图像应该是裁剪好的、只包含单个骑行者的图像。

将输入图像放在 `data/input/` 目录下，或通过参数指定其他输入目录。

### 2. 运行推理

使用以下命令运行推理：

```bash
python inference.py --input data/input --output data/output
```

可选参数：
- `--input`: 输入图像目录或单个图像路径
- `--output`: 输出结果目录
- `--segmentation_model`: 分割模型路径
- `--device`: 计算设备 (cuda 或 cpu)
- `--batch_size`: 批处理大小
- `--workers`: 并行处理的工作线程数
- `--visualize`: 是否生成颜色可视化图像

### 3. 输出格式

系统输出包括：

1. JSON结果文件：每个图像对应一个JSON文件，包含每个部位的颜色和置信度
2. 汇总报告：包含所有处理图像的结果和统计信息
3. 可视化图像（如果启用）：显示每个部位的主要颜色

示例JSON输出：

```json
{
  "helmet": {
    "color": "red",
    "confidence": 0.89
  },
  "upper_body": {
    "color": "black",
    "confidence": 0.95
  },
  "lower_body": {
    "color": "blue",
    "confidence": 0.87
  }
}
```

## 训练模型

如果需要训练自己的模型，可以使用以下步骤：

### 1. 准备训练数据

将训练数据放在 `data/training/` 目录下，数据应包括：
- 骑行者图像
- 分割标注（如果训练分割模型）
- 颜色标注（如果训练颜色分类模型）

### 2. 训练分割模型

```bash
python src/color_extraction/train.py --model_type segmentation --data_dir data/training --epochs 50
```

### 3. 训练颜色分类模型

```bash
python src/color_extraction/train.py --model_type classification --data_dir data/training --epochs 50
```

## 性能优化

为满足每天处理30000+图像的需求，系统已进行以下优化：

1. 多线程并行处理
2. 批处理机制
3. GPU加速（如果可用）
4. 内存优化

您可以通过调整以下参数进一步优化性能：
- `--batch_size`: 增加批处理大小可以提高GPU利用率
- `--workers`: 增加工作线程数可以提高CPU利用率，但需要注意内存消耗

## 故障排除

如果遇到问题，可以查看日志文件（logs目录下）获取详细信息。常见问题包括：

1. CUDA内存不足：尝试减小批处理大小
2. 输入图像读取错误：检查图像格式和路径
3. 模型加载失败：确保模型文件存在并且格式正确

## 进阶使用

### API集成

系统可以作为模块导入到其他Python项目中：

```python
from src.color_extraction.segmentation_model import CyclistSegmentationModel
from src.color_extraction.color_extraction_model import ColorExtractionModel

# 初始化模型
segmentation_model = CyclistSegmentationModel(model_path='models/segmentation_model_best.pth')
extraction_model = ColorExtractionModel(segmentation_model=segmentation_model)

# 提取颜色
result = extraction_model.extract_colors('path/to/image.jpg')
print(result)
``` 