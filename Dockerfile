# 使用官方的Python基础镜像
FROM python:3.11-slim

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_NAME="Salesforce/blip-image-captioning-large"

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载模型和处理器
RUN python3 -c "from transformers import BlipProcessor, BlipForConditionalGeneration; import os; model_name = os.getenv('MODEL_NAME'); processor = BlipProcessor.from_pretrained(model_name); model = BlipForConditionalGeneration.from_pretrained(model_name)"

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 运行FastAPI服务器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]