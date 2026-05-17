import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from face_utils import (
    detect_face,
    get_embedding,
    load_known_faces_encrypted,
    get_base_username,
    save_face_multi_template,
    check_anti_spoofing,
    publish_mqtt_event
)
from main import (
    best_match,
    normalize_embedding,
    safe_name,
    DEFAULT_THRESHOLD,
    MASK_THRESHOLD
)

# Load Env
load_dotenv()

# App Init
app = FastAPI(
    title="DeepLock AI Biometric REST API",
    description="Quantum-Precision FaceUnlock Biometric REST service.",
    version="3.0.0"
)

# Config
FACES_DIR = Path(os.getenv("FACEUNLOCK_FACES_DIR", "faces"))
LOG_FILE = Path(os.getenv("FACEUNLOCK_LOG_FILE", "access_log.csv"))

# API Key Security (Token mặc định nếu chưa cấu hình trong .env)
API_KEY = os.getenv("API_KEY", "deeplock-secret-token-2026")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return api_key

def log_access(username: str, status: str, score: float):
    """Ghi nhật ký đăng nhập đồng bộ."""
    new_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "status": status,
        "score": round(score, 4)
    }
    df = pd.DataFrame([new_entry])
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Chuyển đổi bytes ảnh tải lên thành OpenCV BGR Image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    return img

@app.get("/")
def read_root():
    return {
        "service": "DeepLock Biometric REST Shield",
        "status": "online",
        "version": "3.0.0",
        "endpoints": ["/api/enroll", "/api/verify", "/api/users", "/api/users/{username}"]
    }

@app.post("/api/enroll")
async def enroll_identity(
    username: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Đăng ký định danh mới hỗ trợ Đa mẫu (Multi-template)
    """
    cleaned_name = safe_name(username)
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Invalid username.")
        
    contents = await file.read()
    image = bytes_to_cv2(contents)
    
    # 1. Phát hiện mặt
    box = detect_face(image)
    if box is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
        
    # 2. Trích xuất đặc trưng
    emb = get_embedding(image, box)
    if emb is None:
        raise HTTPException(status_code=400, detail="Failed to extract face embedding.")
        
    # 3. Lưu trữ mã hóa đa mẫu
    norm_emb = normalize_embedding(emb)
    save_idx = save_face_multi_template(cleaned_name, norm_emb, FACES_DIR)
    
    return {
        "status": "success",
        "message": f"Successfully enrolled template {save_idx} for user: {cleaned_name}",
        "username": cleaned_name,
        "template_index": save_idx
    }

@app.post("/api/verify")
async def verify_identity(
    file: UploadFile = File(...),
    security_level: str = Form("High"),
    api_key: str = Depends(verify_api_key)
):
    """
    Xác minh danh tính sinh trắc học thời gian thực
    Tích hợp lá chắn chống giả mạo Liveness (Laplacian & FFT Moire)
    """
    contents = await file.read()
    image = bytes_to_cv2(contents)
    
    # Thiết lập ngưỡng theo cấp độ bảo mật
    threshold = 0.45
    if security_level == "High": threshold = 0.55
    elif security_level == "Ultra": threshold = 0.65
    
    # 1. Tải danh sách mẫu đã đăng ký
    registered = load_known_faces_encrypted(FACES_DIR)
    if not registered:
        raise HTTPException(status_code=400, detail="No registered users in the database.")
        
    # 2. Phát hiện mặt
    box = detect_face(image)
    if box is None:
        return {
            "status": "denied",
            "reason": "No face detected in the image"
        }
        
    # 3. Chống giả mạo (Anti-Spoofing FFT & Laplacian)
    is_real, spoof_reason, spoof_score = check_anti_spoofing(image, box)
    if not is_real:
        log_access("Unknown (SPOOF)", "DENIED", spoof_score)
        return {
            "status": "denied",
            "reason": spoof_reason,
            "spoof_score": round(spoof_score, 4)
        }
        
    # 4. Trích xuất đặc trưng và so khớp
    emb = get_embedding(image, box)
    if emb is None:
        return {
            "status": "denied",
            "reason": "Failed to extract face embedding"
        }
        
    name, score = best_match(emb, registered)
    
    # 5. Phê duyệt hoặc từ chối
    if score >= threshold:
        log_access(name, "GRANTED", score)
        # Gửi sự kiện MQTT về Smart Home
        publish_mqtt_event(name, "GRANTED")
        return {
            "status": "granted",
            "username": name,
            "score": round(score, 4),
            "spoof_score": round(spoof_score, 4),
            "message": f"Welcome {name}! Access granted."
        }
    else:
        log_access("Unknown (LOW_SCORE)", "DENIED", score)
        return {
            "status": "denied",
            "reason": "Low confidence match",
            "score": round(score, 4),
            "threshold": threshold
        }

@app.get("/api/users")
def list_users(api_key: str = Depends(verify_api_key)):
    """
    Liệt kê danh sách người dùng và số lượng mẫu đăng ký tương ứng
    """
    registered = load_known_faces_encrypted(FACES_DIR)
    grouped_users = {}
    for full_name in list(registered.keys()):
        base = get_base_username(full_name)
        grouped_users[base] = grouped_users.get(base, 0) + 1
    return {
        "status": "success",
        "total_users": len(grouped_users),
        "users": grouped_users
    }

@app.delete("/api/users/{username}")
def delete_user(username: str, api_key: str = Depends(verify_api_key)):
    """
    Xóa toàn bộ các tệp đặc trưng của người dùng này khỏi đĩa cứng
    """
    cleaned_name = safe_name(username)
    deleted_any = False
    for suffix in ["", "_0", "_1", "_2"]:
        for ext in [".npy", ".enc"]:
            p = FACES_DIR / f"{cleaned_name}{suffix}{ext}"
            if p.exists():
                p.unlink()
                deleted_any = True
                
    if deleted_any:
        return {
            "status": "success",
            "message": f"Deep-deleted all templates for user: {cleaned_name}"
        }
    else:
        raise HTTPException(status_code=404, detail="User not found.")
