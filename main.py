import io
import os
import re
import typing

import requests
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# 加载模型和处理器
model_name = os.getenv('MODEL_NAME', 'Salesforce/blip-image-captioning-base')
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)


class CommonResponse(BaseModel):
    code: int = 500
    message: str = '调用成功'
    data: typing.Any = []


@app.post("/api/predict")
async def get_image_caption(
        file: UploadFile = File(...),
        target_lang: str = Form(None)
):
    # 读取图像文件
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 处理图像
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)

    # 生成描述
    caption = processor.decode(output[0], skip_special_tokens=True)

    # 后处理步骤：移除不合理的前缀
    caption = clean_caption(caption)

    # 支持的语言列表
    supported_langs = [
        "af", "sq", "am", "ar", "hy", "az", "eu", "be", "bn", "bs", "bg", "ca", "ceb", "ny",
        "zh", "zh-TW", "co", "hr", "cs", "da", "nl", "en", "eo", "et", "tl", "fi", "fr", "fy",
        "gl", "ka", "de", "el", "gu", "ht", "ha", "haw", "he", "hi", "hmn", "hu", "is", "ig",
        "id", "ga", "it", "ja", "jw", "kn", "kk", "km", "rw", "ko", "ku", "ky", "lo", "la",
        "lv", "lt", "lb", "mk", "mg", "ms", "ml", "mt", "mi", "mr", "mn", "my", "ne", "no",
        "or", "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sm", "gd", "sr", "st", "sn", "sd",
        "si", "sk", "sl", "so", "es", "su", "sw", "sv", "tg", "ta", "tt", "te", "th", "tr",
        "tk", "uk", "ur", "ug", "uz", "vi", "cy", "xh", "yi", "yo", "zu"
    ]
    if target_lang and target_lang in supported_langs:
        # 翻译成特定语言
        caption = translate(caption, target_lang)

    return CommonResponse(code=200, data={'result': caption, })


# 移除多余的内容
def clean_caption(caption: str) -> str:
    # 使用正则表达式移除不合理的前缀
    cleaned_caption = re.sub(r'\b(arafed|araffe|araff)\b', '', caption)
    return cleaned_caption.strip()


# 翻译工具
def translate(text: str, target_lang: str) -> str:
    with requests.get('https://lingva.thedaviddelta.com/api/v1/auto/' + target_lang + '/' + text) as response:
        text = ''
        if response.status_code == 200:
            data = response.json()
            text = data['translation']
        return text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
