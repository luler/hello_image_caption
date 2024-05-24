# 使用官方的Python基础镜像
FROM python:3.11-slim

# 复制应用代码
COPY . /app

# 设置工作目录
WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载模型和处理器
RUN python3 -c "from transformers import BlipProcessor, BlipForConditionalGeneration; model_name = 'Salesforce/blip-image-captioning-base'; processor = BlipProcessor.from_pretrained(model_name); model = BlipForConditionalGeneration.from_pretrained(model_name)"

# 暴露端口
EXPOSE 8000

# 运行FastAPI服务器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]