import base64
import io
import cv2
import numpy as np
from PIL import Image

def decode_image_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image buffer")
    return image

def decode_image_base64(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    decoded = base64.b64decode(b64_string)
    return decode_image_bytes(decoded)

def encode_image_base64(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError("Failed to encode image to buffer")
    encoded = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{encoded}"

def resize_keep_aspect(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def compute_difference_heatmap(img_a: np.ndarray, img_b: np.ndarray, size=(160, 160)) -> np.ndarray:
    a_resized = cv2.resize(img_a, size)
    b_resized = cv2.resize(img_b, size)
    
    gray_a = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY) if len(a_resized.shape) == 3 else a_resized
    gray_b = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY) if len(b_resized.shape) == 3 else b_resized
    
    diff = cv2.absdiff(gray_a, gray_b)
    diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    
    blended = cv2.addWeighted(b_resized if len(b_resized.shape) == 3 else cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR), 0.55, heatmap, 0.45, 0)
    return blended
