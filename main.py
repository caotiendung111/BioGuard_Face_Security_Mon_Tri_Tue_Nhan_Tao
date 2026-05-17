import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine

from face_utils import (
    FaceBox,
    check_blink,
    crop_upper_face,
    detect_face,
    get_embedding,
    get_face_landmarks,
    is_wearing_mask,
    load_known_faces_encrypted,
    save_face_encrypted,
    get_base_username,
    save_face_multi_template,
)


BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
WINDOW_NAME = "FaceUnlock"
DEFAULT_THRESHOLD = 0.50
MASK_THRESHOLD = 0.42
BLINK_FRAMES = 10
UNLOCK_HOLD_SECONDS = 3.0
RECONNECT_DELAY_SECONDS = 1.0


def safe_name(name: str) -> str:
    """Lam sach ten de dung lam ten file .npy."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned or "user"


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Chuan hoa vector truoc khi luu/so sanh."""
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


def load_embeddings() -> Dict[str, np.ndarray]:
    """Đọc và giải mã toàn bộ khuôn mặt đã đăng ký từ faces/."""
    return load_known_faces_encrypted(FACES_DIR)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """SciPy cosine la distance, nen similarity = 1 - distance."""
    return 1.0 - float(cosine(normalize_embedding(a), normalize_embedding(b)))


def best_match(embedding: np.ndarray, registered: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
    """Tìm embedding đã đăng ký có điểm similarity cao nhất, tự động tách tên người dùng gốc."""
    best_base_name = None
    best_score = -1.0
    for full_name, known in registered.items():
        base_name = get_base_username(full_name)
        score = cosine_similarity(embedding, known)
        if score > best_score:
            best_base_name = base_name
            best_score = score
    return best_base_name, best_score


def build_stream_url(args) -> str:
    """Lay URL stream tu tham so, bien moi truong hoac hoi nguoi dung."""
    if args.url:
        return args.url

    env_url = os.getenv("FACEUNLOCK_CAMERA_URL")
    if env_url:
        return env_url

    ip = args.ip or os.getenv("FACEUNLOCK_PHONE_IP")
    while not ip:
        ip = input("Nhap IP dien thoai (vd 192.168.1.5): ").strip()

    return f"http://{ip}:{args.port}/video"


def open_stream(source) -> cv2.VideoCapture:
    """Mo stream tu IP URL hoac camera index USB."""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def draw_ui(frame, box: Optional[FaceBox], status: str, color, detail: str = ""):
    """Ve bounding box, trang thai va phim tat."""
    if box is not None:
        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if detail:
        cv2.putText(frame, detail, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.putText(
        frame,
        "r: register | q: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def waiting_frame(message: str) -> np.ndarray:
    """Tao frame den khi mat ket noi stream."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, message, (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def collect_liveness(cap: cv2.VideoCapture, box: FaceBox) -> bool:
    """Thu 10 frame tu IP stream de kiem tra chop mat."""
    history = []

    for _ in range(BLINK_FRAMES):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = cv2.flip(frame, 1)
        landmarks = get_face_landmarks(frame)
        history.append(landmarks)
        draw_ui(frame, box, "BLINK NOW", (0, 255, 255), "Look at phone camera and blink once")
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    return check_blink(history)


def register_face(frame: np.ndarray, box: FaceBox, landmarks, registered: Dict[str, np.ndarray]):
    """Đăng ký người dùng mới từ frame hiện tại bằng mã hóa sinh trắc học Fernet."""
    if is_wearing_mask(landmarks):
        print("Đăng ký thất bại: Hãy tháo khẩu trang khi đăng ký.")
        return

    name = safe_name(input("Nhập tên người dùng: "))
    target = FACES_DIR / f"{name}.npy"
    if target.exists():
        answer = input(f"{target.name} đã tồn tại. Ghi đè? [y/N]: ").strip().lower()
        if answer != "y":
            print("Đã hủy đăng ký.")
            return

    embedding = get_embedding(frame, box)
    if embedding is None:
        print("Không tạo được embedding, vui lòng thử lại.")
        return

    # Chuẩn hóa Vector
    embedding = normalize_embedding(embedding)
    
    # Mã hóa và lưu trữ bảo mật hỗ trợ đa mẫu (Multi-template)
    save_idx = save_face_multi_template(name, embedding, FACES_DIR)
    print(f"Da dang ky va ma hoa an toan file faces/{name}_{save_idx}.npy")
    
    # Nạp lại toàn bộ danh sách để đồng bộ hệ thống
    registered.clear()
    registered.update(load_embeddings())


def main():
    # Tải biến môi trường
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="FaceUnlock dung IVcam/IP Camera/USB camera ao")
    parser.add_argument("--camera", type=int, default=None, help="Dung camera index, vd 1 cho IVcam USB")
    parser.add_argument("--url", default="", help="URL stream day du, vd http://192.168.1.5:8080/video")
    parser.add_argument("--ip", default="", help="IP dien thoai, vd 192.168.1.5")
    parser.add_argument("--port", default="8080", help="Port stream, mac dinh 8080")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Nguong full-face")
    args = parser.parse_args()

    source = args.camera if args.camera is not None else build_stream_url(args)
    registered = load_embeddings()
    print(f"Dang ket noi camera: {source}")
    print(f"Da nap {len(registered)} khuon mat da dang ky.")

    cap = open_stream(source)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    unlocked_until = 0.0
    last_reconnect = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            now = time.time()
            if now - last_reconnect > RECONNECT_DELAY_SECONDS:
                cap.release()
                cap = open_stream(source)
                last_reconnect = now

            placeholder = waiting_frame("Dang cho ket noi camera...")
            cv2.imshow(WINDOW_NAME, placeholder)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        frame = cv2.flip(frame, 1)
        box = detect_face(frame)
        landmarks = get_face_landmarks(frame) if box is not None else None

        status = "LOCKED"
        detail = "No face detected"
        color = (0, 0, 255)

        if time.time() < unlocked_until:
            status = "UNLOCKED"
            detail = "Access granted"
            color = (0, 200, 0)
        elif box is not None:
            if not registered:
                detail = "No faces registered"
            else:
                wearing_mask = is_wearing_mask(landmarks)
                if wearing_mask:
                    face_image, _ = crop_upper_face(frame, landmarks)
                    embedding = get_embedding(face_image)
                    threshold = MASK_THRESHOLD
                else:
                    embedding = get_embedding(frame, box)
                    threshold = args.threshold

                if embedding is None:
                    detail = "Cannot extract embedding"
                else:
                    name, score = best_match(embedding, registered)
                    detail = f"{'MASK' if wearing_mask else 'NO MASK'} | best={name} | score={score:.2f}"

                    if score > threshold:
                        if collect_liveness(cap, box):
                            status = "UNLOCKED"
                            detail = f"Access granted: {name}"
                            color = (0, 200, 0)
                            unlocked_until = time.time() + UNLOCK_HOLD_SECONDS
                        else:
                            status = "LIVENESS_FAIL"
                            detail = "Blink not detected"

        draw_ui(frame, box, status, color, detail)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            if box is None:
                print("Khong thay khuon mat de dang ky.")
            else:
                register_face(frame, box, landmarks, registered)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
