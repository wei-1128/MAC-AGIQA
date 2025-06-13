# utils.py

import base64
import os
from PIL import Image
from io import BytesIO
import logging


def load_image_as_base64(image_path, resize=None):
    """
    加载图像并转换为 base64 编码字符串（用于上传接口）
    
    参数：
    - image_path: str，图像路径
    - resize: tuple(w, h)，可选，是否重新调整图像尺寸

    返回：
    - base64_str: str
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    if resize:
        img = img.resize(resize)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str


def pretty_print_dict(d, title=None):
    """
    美观地打印字典
    """
    if title:
        print(f"\n==== {title} ====")
    for k, v in d.items():
        print(f"{k}: {v}")


def init_logger(name="mac_agiqa", level=logging.INFO):
    """
    初始化日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
