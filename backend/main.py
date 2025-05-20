from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.box_detector import Detector
import numpy as np
import cv2
import base64
from backend import db_utils  # Import các hàm từ db_utils.py

app = FastAPI()
detector = Detector()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoint for processing frames
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    # Đọc nội dung file ảnh / Read image file content
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Xử lý khung hình / Process the frame
    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)
    
    # Trả về kết quả / Return results
    face_boxes_for_response = []
    for (coords, conf, emotion, embedding) in face_boxes:
        if embedding:
            similar_faces = db_utils.find_similar_faces(embedding)  # Gọi hàm từ db_utils
            face_names = [face["name"] for face in similar_faces]
            face_boxes_for_response.append({
                "coords": coords,
                "confidence": conf,
                "emotion": emotion,
                "similar_faces": face_names,
            })
        else:
            face_boxes_for_response.append({
                "coords": coords,
                "confidence": conf,
                "emotion": emotion,
                "similar_faces": [],
            })
    return {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf, "action": action}
            for (coords, conf, action) in person_boxes
        ],
        "face_boxes": face_boxes_for_response,
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Mount the static files directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Serve index.html at the root
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))