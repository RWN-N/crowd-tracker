import io
import cv2
import numpy as np
from base64 import b64decode, b64encode
from PIL import Image

def bytes_to_image(image_str: str) -> np.ndarray:
    _, image_bytes = image_str.split(',')  # header,bytes
    image_bytes = b64decode(image_bytes)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, flags=1)
    return image

def image_to_bytes(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array, 'RGBA')
    buf = io.BytesIO()
    image.save(buf, format='png')
    image_bytes = f"data:image/png;base64,{b64encode(buf.getvalue()).decode('utf-8')}"
    return image_bytes