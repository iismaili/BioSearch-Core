# BioSearch-Core 👁️

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end interactive computer vision pipeline visualizer and biometric face recognition suite. Designed with a sleek, glassmorphic dark-mode interface, **BioSearch-Core** guides users step-by-step through fundamental digital image processing transformations and performs deep-learning-based face identification across multi-person scenes.

---

## Key Features

- **11-Stage Interactive Image Processing Pipeline**:
  Inspect every intermediate stage of image analysis with side-by-side technical descriptions, mathematical formulations, and computed telemetry metrics.
- **Biometric Landmark Extraction & Alignment**:
  5-point facial landmark detection (pupils, nose tip, mouth corners) with affine rotation alignment to normalize facial tilt.
- **Deep Feature Representation (128D Embeddings)**:
  Extracts unit-sphere normalized feature vectors via SFace and computes directional gradient maps using HOG (Histogram of Oriented Gradients).
- **Multi-Person Face Search & Matching (Stage 2)**:
  Upload any group scene or crowd image to automatically locate and verify the target person using cosine similarity distance and neon-accented bounding boxes.
- **Difference Heatmap & Candidate Telemetry**:
  Visualizes anatomical intensity differences and ranks all candidate faces with confidence scores, L2 norms, and cosine metrics.
- **Embedded One-Click Test Suite**:
  Ships with pre-bundled studio portraits and multi-person test scenes for instantaneous verification.

---

## Image Processing Pipeline Breakdown

| Stage # | Stage Name | Category | Description |
|---|---|---|---|
| **01** | Raw Input Normalization | Input Analysis | Resolution standardization, BGR color space verification, and dynamic aspect scaling. |
| **02** | Grayscale Conversion | Color Space | Luminance reduction using Rec. 601 Luma ($Y = 0.299R + 0.587G + 0.114B$), reducing computation overhead. |
| **03** | Gaussian Smoothing | Spatial Filtering | 2D Gaussian kernel convolution ($\sigma=1.2$) removing high-frequency sensor noise. |
| **04** | Contrast Enhancement (CLAHE) | Histogram Equalization | Contrast Limited Adaptive Histogram Equalization over local $8\times8$ grids to normalize shadows. |
| **05** | Edge Detection (Canny & Sobel) | Boundary Analysis | Gradient magnitude ($G = \sqrt{G_x^2 + G_y^2}$), non-maximum suppression, and hysteresis thresholding. |
| **06** | Otsu Binarization | Segmentation | Automatic between-class variance maximization separating foreground biometric structures. |
| **07** | Morphological Operations | Mathematical Morphology | Consecutive opening and closing using elliptical kernels to eliminate speckle noise and close contours. |
| **08** | Laplacian & Sharpening | Frequency Analysis | Second-derivative operator ($\nabla^2 f$) computing focus variance and enhancing micro-textures. |
| **09** | Face ROI Detection | Object Localization | Deep YuNet anchor-free convolutional detector extracting face bounding box coordinates $[x, y, w, h]$. |
| **10** | 5-Point Landmark Alignment | Biometric Geometry | Affine transformation based on inter-ocular angle $\theta = \arctan2(\Delta y, \Delta x)$ for canonical $112\times112$ scaling. |
| **11** | Feature Extraction (HOG & 128D) | Embedding Space | Gradient orientation histograms combined with 128-dimensional deep feature representation. |

---

## Architecture Overview

```
BioSearch-Core/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server, REST routes & static file handlers
│   ├── pipeline.py          # 11-stage image processing pipeline & telemetry
│   ├── matcher.py           # Multi-face detection, distance matching & annotator
│   └── utils.py             # Base64 enc/dec, image rescaling, difference heatmaps
├── models/
│   ├── face_detection_yunet.onnx      # Lightweight YuNet face detector
│   └── face_recognition_sface.onnx    # SFace 128D face feature extractor
├── sample_data/
│   ├── sample_person.jpg             # Reference person portrait
│   ├── sample_group_match.jpg        # Multi-person scene containing target
│   └── sample_group_nomatch.jpg      # Multi-person scene without target
├── static/
│   ├── index.html           # Modern semantic HTML5 interface
│   ├── style.css            # Glassmorphic responsive dark theme
│   └── app.js               # Reactive client controller & state manager
├── download_models.py       # Automatic ONNX weights downloader
├── requirements.txt         # Project dependencies
├── run.py                   # Application launch entrypoint
└── start.sh                 # Environment setup and run script
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/BioSearch-Core.git
   cd BioSearch-Core
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download ONNX models**:
   ```bash
   python download_models.py
   ```
   *(Note: The startup script automatically downloads models if they are missing).*

### Running the Application

Launch the server with the runner script:

```bash
python run.py
```

Or via bash:
```bash
./start.sh
```

Navigate to [http://localhost:8000](http://localhost:8000) in your web browser.

---

## API Reference

### 1. `POST /api/analyze-stage1`
Executes the 11-stage computer vision pipeline on the input image.
- **Request**: Multipart Form Data (`file`) or URL Encoded (`image_base64`).
- **Response**:
  ```json
  {
    "success": true,
    "has_face": true,
    "stages": [
      {
        "id": "raw_input",
        "step_number": 1,
        "title": "Orijinal Görüntü & Ön Normalizasyon",
        "category": "Girdi Analizi",
        "image_data": "data:image/jpeg;base64,...",
        "what_is_it": "...",
        "why_it_matters": "...",
        "technical_details": "...",
        "metrics": { ... }
      }
    ],
    "cropped_face_b64": "data:image/jpeg;base64,...",
    "feature_vector": [0.012, -0.045, ...]
  }
  ```

### 2. `POST /api/match-stage2`
Performs face verification and similarity matching against the reference vector.
- **Request Body**:
  ```json
  {
    "image_base64": "data:image/jpeg;base64,...",
    "reference_vector": [ ... ],
    "reference_crop_b64": "data:image/jpeg;base64,...",
    "threshold": 0.363
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "match_found": true,
    "total_faces_detected": 5,
    "best_match": {
      "index": 2,
      "similarity_pct": 92.4,
      "cosine_similarity": 0.7312,
      "l2_distance": 0.733,
      "is_match": true,
      "crop_b64": "..."
    },
    "annotated_image_b64": "data:image/jpeg;base64,...",
    "diff_heatmap_b64": "data:image/jpeg;base64,..."
  }
  ```

### 3. `GET /api/sample-images`
Returns pre-bundled high-resolution sample images for immediate UI testing.

---

## License

This project is licensed under the [MIT License](LICENSE).
