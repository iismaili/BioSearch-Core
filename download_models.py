import os
import urllib.request

MODELS = {
    "face_detection_yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def ensure_models(target_dir="models"):
    os.makedirs(target_dir, exist_ok=True)
    for name, url in MODELS.items():
        dest = os.path.join(target_dir, name)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            print(f"Saved {name} ({os.path.getsize(dest)} bytes)")
        else:
            print(f"Found existing model: {name}")

if __name__ == "__main__":
    ensure_models()
