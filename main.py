import io
import typing

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# 加载模型和处理器
model_name = "Salesforce/blip-image-captioning-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)


class CommonResponse(BaseModel):
    code: int = 500
    message: str = '调用成功'
    data: typing.Any = []


@app.post("/api/predict")
async def get_image_caption(file: UploadFile = File(...)):
    # 读取图像文件
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 处理图像
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)

    # 生成描述
    caption = processor.decode(output[0], skip_special_tokens=True)

    return CommonResponse(code=200, data={'result': caption, })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
