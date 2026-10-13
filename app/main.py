import os
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.utils import decode_image_bytes, decode_image_base64, encode_image_base64
from app.pipeline import VisionPipeline
from app.matcher import FaceMatcher

app = FastAPI(
    title="BioSearch-Core - Image Processing & Face Recognition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

pipeline = VisionPipeline(models_dir=MODELS_DIR)
matcher = FaceMatcher(models_dir=MODELS_DIR)

from typing import Optional, List, Union

class MatchStage2Request(BaseModel):
    image_base64: str
    reference_vector: Union[List[float], List[List[float]]]
    reference_crop_b64: Optional[str] = None
    threshold: Optional[float] = 0.363

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "detector_loaded": pipeline.detector is not None,
        "recognizer_loaded": pipeline.recognizer is not None
    }

@app.post("/api/analyze-stage1")
async def analyze_stage1(file: Optional[UploadFile] = File(None), image_base64: Optional[str] = Form(None)):
    try:
        if file is not None:
            content = await file.read()
            img = decode_image_bytes(content)
        elif image_base64:
            img = decode_image_base64(image_base64)
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        result = pipeline.process_image(img)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match-stage2")
async def match_stage2(payload: MatchStage2Request):
    try:
        img = decode_image_base64(payload.image_base64)
        if payload.threshold is not None:
            matcher.cosine_threshold = float(payload.threshold)
        result = matcher.search_person(
            search_img=img,
            reference_vector=payload.reference_vector,
            reference_crop_b64=payload.reference_crop_b64
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sample-images")
async def list_sample_images():
    samples = []
    sample_files = [
        {
            "id": "target_person",
            "filename": "sample_person.jpg",
            "title": "Referans Kişi (Hedef)",
            "description": "Net stüdyo aydınlatmalı, gözlüklü genç erkek portresi",
            "type": "reference"
        },
        {
            "id": "group_match",
            "filename": "sample_group_match.jpg",
            "title": "Grup Fotoğrafı (Kişi Mevcut)",
            "description": "3 iş arkadaşının ofiste çekilmiş fotoğrafı (hedef kişi ortada)",
            "type": "search"
        },
        {
            "id": "group_nomatch",
            "filename": "sample_group_nomatch.jpg",
            "title": "Grup Fotoğrafı (Kişi Yok)",
            "description": "Toplantı masasındaki farklı 3 çalışma arkadaşı",
            "type": "search"
        }
    ]
    
    for item in sample_files:
        path = os.path.join(SAMPLE_DIR, item["filename"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                content = f.read()
            import cv2
            img = decode_image_bytes(content)
            item["image_data"] = encode_image_base64(img)
            samples.append(item)
            
    return JSONResponse(content=samples)

@app.get("/")
async def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "BioSearch-Core API is running"}

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
