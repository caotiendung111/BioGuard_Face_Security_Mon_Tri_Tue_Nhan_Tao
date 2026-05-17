import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import face_recognition
import mediapipe as mp
import numpy as np


# face_recognition dung format bbox: (top, right, bottom, left)
FaceBox = Tuple[int, int, int, int]
Point3D = Tuple[float, float, float]

MASK_NOSE = (1, 2, 98, 327)
MASK_MOUTH = (13, 14, 17)
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
CHIN = 152


try:
    _mp_face_mesh = mp.solutions.face_mesh
    _mp_face_detection = mp.solutions.face_detection
except AttributeError:
    _mp_face_mesh = None
    _mp_face_detection = None

_FACE_MESH = None
_FACE_DETECTOR = None


def _get_mesh():
    global _FACE_MESH
    if _mp_face_mesh is None:
        return None
    if _FACE_MESH is None:
        _FACE_MESH = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _FACE_MESH


def _get_detector():
    global _FACE_DETECTOR
    if _mp_face_detection is None:
        return None
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = _mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    return _FACE_DETECTOR


def detect_face(image: np.ndarray) -> Optional[FaceBox]:
    """Phat hien khuon mat bang MediaPipe (nhanh hon dlib)."""
    if image is None or image.size == 0:
        return None

    # Tăng cường ánh sáng yếu (Night Shift AI)
    image = enhance_low_light(image)

    detector = _get_detector()
    if detector is None:
        # Fallback sang dlib neu mediapipe loi
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return None
        return max(locations, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        return None

    # Lay face lon nhat
    best_detection = max(
        results.detections,
        key=lambda d: d.location_data.relative_bounding_box.width
        * d.location_data.relative_bounding_box.height,
    )
    bbox = best_detection.location_data.relative_bounding_box
    
    # Chuyen doi sang format (top, right, bottom, left)
    top = int(bbox.ymin * height)
    left = int(bbox.xmin * width)
    bottom = int((bbox.ymin + bbox.height) * height)
    right = int((bbox.xmin + bbox.width) * width)
    
    return (max(0, top), min(width, right), min(height, bottom), max(0, left))



def get_embedding(image: np.ndarray, bbox: Optional[FaceBox] = None) -> Optional[np.ndarray]:
    """Trich xuat embedding 128 chieu bang model pre-trained cua face_recognition."""
    if image is None or image.size == 0:
        return None

    # Tăng cường ánh sáng yếu (Night Shift AI)
    image = enhance_low_light(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if bbox is None:
        height, width = rgb.shape[:2]
        locations = face_recognition.face_locations(rgb, model="hog")
        bbox_list = locations if locations else [(0, width, height, 0)]
    else:
        bbox_list = [bbox]

    encodings = face_recognition.face_encodings(rgb, known_face_locations=bbox_list)
    if not encodings:
        return None
    return np.asarray(encodings[0], dtype=np.float32)


def get_face_landmarks(image: np.ndarray) -> Optional[List[Point3D]]:
    """Lay 468 landmarks bang MediaPipe Face Mesh; fallback sang dlib neu can."""
    if image is None or image.size == 0:
        return None

    mesh = _get_mesh()
    if mesh is None:
        return _get_dlib_landmarks(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    return [(lm.x, lm.y, lm.z) for lm in result.multi_face_landmarks[0].landmark]


def _get_dlib_landmarks(image: np.ndarray) -> Optional[List[Point3D]]:
    """Fallback: map mot so moc dlib vao index MediaPipe can dung."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_landmarks(rgb)
    if not faces:
        return None

    height, width = image.shape[:2]
    data = faces[0]
    landmarks: List[Point3D] = [(0.0, 0.0, 0.0) for _ in range(468)]

    def put(index: int, point):
        landmarks[index] = (point[0] / width, point[1] / height, 0.0)

    for index, point in zip(LEFT_EYE, data.get("left_eye", [])[:6]):
        put(index, point)
    for index, point in zip(RIGHT_EYE, data.get("right_eye", [])[:6]):
        put(index, point)
    for index, point in zip(MASK_NOSE, (data.get("nose_bridge", []) + data.get("nose_tip", []))[:4]):
        put(index, point)
    for index, point in zip(MASK_MOUTH, (data.get("top_lip", []) + data.get("bottom_lip", []))[:3]):
        put(index, point)

    chin_points = data.get("chin", [])
    if chin_points:
        put(CHIN, chin_points[len(chin_points) // 2])

    return landmarks


def _valid_point(point: Point3D) -> bool:
    x, y, _ = point
    return 0.0 < x < 1.0 and 0.0 < y < 1.0


def _dist(a: Point3D, b: Point3D) -> float:
    return math.dist((a[0], a[1]), (b[0], b[1]))


def is_wearing_mask(landmarks: Optional[Sequence[Point3D]]) -> bool:
    """
    Phat hien khau trang don gian:
    - Neu nhieu diem mui/moi khong hop le thi coi la bi che.
    - Neu khoang mui-cam hoac moi-cam bat thuong nho thi coi la co khau trang.
    """
    if landmarks is None or len(landmarks) <= CHIN:
        return False

    mask_indices = MASK_NOSE + MASK_MOUTH
    visible = sum(1 for idx in mask_indices if idx < len(landmarks) and _valid_point(landmarks[idx]))
    if visible < len(mask_indices) * 0.6:
        return True

    nose_tip = landmarks[1]
    upper_lip = landmarks[13]
    chin = landmarks[CHIN]
    if not (_valid_point(nose_tip) and _valid_point(upper_lip) and _valid_point(chin)):
        return False

    nose_chin = _dist(nose_tip, chin)
    lip_chin = _dist(upper_lip, chin)

    # Nguong thuc nghiem tren toa do normalized. Khau trang thuong lam vung mui/moi kem on dinh.
    return nose_chin < 0.12 or lip_chin < 0.06


def crop_upper_face(image: np.ndarray, landmarks: Optional[Sequence[Point3D]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop vung upper-face tu dinh tran den duoi mat, tra ve (crop, (x, y, w, h))."""
    if image is None or image.size == 0:
        return image, (0, 0, 0, 0)

    height, width = image.shape[:2]
    if landmarks is None:
        return image, (0, 0, width, height)

    valid_points = [(x * width, y * height) for x, y, _ in landmarks if 0.0 < x < 1.0 and 0.0 < y < 1.0]
    if not valid_points:
        return image, (0, 0, width, height)

    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    eye_ys = [
        landmarks[idx][1] * height
        for idx in LEFT_EYE + RIGHT_EYE
        if idx < len(landmarks) and _valid_point(landmarks[idx])
    ]

    x1 = max(0, int(min(xs) - 20))
    x2 = min(width, int(max(xs) + 20))
    y1 = max(0, int(min(ys) - 35))
    y2 = min(height, int(max(eye_ys) + 45)) if eye_ys else min(height, int(y1 + (max(ys) - min(ys)) * 0.55))

    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, width, height)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _ear(landmarks: Sequence[Point3D], eye_indices: Tuple[int, int, int, int, int, int]) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in eye_indices]
    vertical_1 = _dist(p2, p6)
    vertical_2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-8:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def check_blink(
    landmarks_list: Sequence[Sequence[Point3D]],
    low_threshold: float = 0.25,
    high_threshold: float = 0.30,
) -> bool:
    """Phat hien chop mat: EAR < 0.25 roi tang lai > 0.3 trong chuoi 10 frame."""
    saw_closed = False

    for landmarks in landmarks_list:
        if landmarks is None:
            continue
        if len(landmarks) <= max(max(LEFT_EYE), max(RIGHT_EYE)):
            continue

        left = _ear(landmarks, LEFT_EYE)
        right = _ear(landmarks, RIGHT_EYE)
        ear = (left + right) / 2.0

        if ear < low_threshold:
            saw_closed = True
        elif saw_closed and ear > high_threshold:
            return True

    return False


def get_head_pose(landmarks: List[Point3D]) -> Tuple[float, float, float]:
    """Uoc luong huong dau (pitch, yaw, roll) tu landmarks."""
    if not landmarks or len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    # Lay cac diem moc chinh
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Map vao toa do 2D (don gian hoa)
    pitch = (nose_tip[1] - (left_eye[1] + right_eye[1]) / 2) * 100
    yaw = (nose_tip[0] - (left_eye[0] + right_eye[0]) / 2) * 100
    roll = (right_eye[1] - left_eye[1]) * 100

    return pitch, yaw, roll


def check_liveness_v2(history: List[List[Point3D]]) -> Tuple[bool, str]:
    """Kiem tra liveness nang cao: chop mat + quay dau."""
    if len(history) < 5:
        return False, "Collecting data..."

    saw_blink = check_blink(history)
    yaws = [get_head_pose(lm)[1] for lm in history if lm]
    yaw_diff = max(yaws) - min(yaws) if yaws else 0
    
    if not saw_blink:
        return False, "Please blink"
    if yaw_diff < 2.0:
        return False, "Turn head slightly"
        
    return True, "Verified"


def draw_face_mesh(image: np.ndarray, landmarks: List[Point3D]):
    """Ve mesh toi uu de khong gay lag va fix loi line tu goc man hinh."""
    height, width = image.shape[:2]
    
    # Ve cac diem vung mat, moi de co cam giac AI nhung khong nang may
    def draw_list(indices, color=(0, 255, 255), closed=True):
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                p = landmarks[idx]
                # Bo qua cac diem loi (0,0) hoac ngoai man hinh
                if 0.001 < p[0] < 0.999 and 0.001 < p[1] < 0.999:
                    pts.append((int(p[0] * width), int(p[1] * height)))
        
        if len(pts) > 1:
            for i in range(len(pts) - (0 if closed else 1)):
                cv2.line(image, pts[i], pts[(i + 1) % len(pts)], color, 1)

    # 1. Mat trai & phai
    draw_list(LEFT_EYE, color=(0, 255, 255))
    draw_list(RIGHT_EYE, color=(0, 255, 255))
    
    # 2. Moi
    draw_list(MASK_MOUTH, color=(0, 255, 255))
    
    # 3. Song mui (Moi them de thay full mat)
    draw_list([168, 6, 197, 195, 5], color=(0, 255, 255), closed=False)
    
    # 4. Vien khuon mat (Face Oval - Tao hieu ung quat full mat)
    draw_list([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], color=(0, 255, 255))


def enhance_low_light(image: np.ndarray, brightness_threshold: float = 75.0) -> np.ndarray:
    """Tăng cường độ tương phản và độ sáng bằng CLAHE nếu ảnh quá tối."""
    if image is None or image.size == 0:
        return image
    
    # Tính toán độ sáng trung bình thông qua ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    
    if avg_brightness < brightness_threshold:
        # Chuyển đổi sang hệ màu LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) lên kênh L (Lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Ghép các kênh lại và trả về hệ màu BGR ban đầu
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
    return image


def load_key() -> bytes:
    """Tải hoặc tự động tạo và lưu trữ khóa mã hóa Fernet vào tệp .env."""
    env_path = Path(__file__).resolve().parent / ".env"
    
    # Load dotenv để cập nhật biến môi trường nếu có sẵn
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    
    key_str = os.getenv("ENCRYPTION_KEY")
    if not key_str:
        # Tạo khóa Fernet mới ngẫu nhiên
        key = Fernet.generate_key()
        
        # Ghi thủ công để đảm bảo hoạt động an toàn và tương thích trên mọi HĐH
        if env_path.exists():
            try:
                content = env_path.read_text(encoding="utf-8")
                lines = content.splitlines()
            except Exception:
                lines = []
            
            new_lines = []
            found = False
            for line in lines:
                if line.strip().startswith("ENCRYPTION_KEY="):
                    new_lines.append(f"ENCRYPTION_KEY={key.decode()}")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"ENCRYPTION_KEY={key.decode()}")
            
            env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        else:
            env_path.write_text(f"ENCRYPTION_KEY={key.decode()}\n", encoding="utf-8")
            
        # Nạp lại biến môi trường
        load_dotenv(dotenv_path=env_path)
        return key
        
    return key_str.encode()


def encrypt_embedding(embedding: np.ndarray, key: bytes) -> bytes:
    """Mã hóa vector embedding bằng Fernet sau khi đã đóng gói bằng pickle."""
    if embedding is None:
        return b""
    serialized = pickle.dumps(embedding)
    f = Fernet(key)
    return f.encrypt(serialized)


def decrypt_embedding(encrypted_data: bytes, key: bytes) -> np.ndarray:
    """Giải mã dữ liệu thô đã mã hóa và tái tạo lại vector numpy embedding."""
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return pickle.loads(decrypted_data)


def save_face_encrypted(name: str, embedding: np.ndarray, faces_dir="faces"):
    """Mã hóa và lưu trữ đặc trưng khuôn mặt vào thư mục dưới dạng file .npy (đã mã hóa bảo mật)."""
    p_dir = Path(faces_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    
    key = load_key()
    encrypted_bytes = encrypt_embedding(embedding, key)
    
    target_path = p_dir / f"{name}.npy"
    with open(target_path, "wb") as f:
        f.write(encrypted_bytes)
    print(f"[SECURE] Da ma hoa va luu tru thanh cong khuon mat: {name}.npy")


def load_known_faces_encrypted(faces_dir="faces") -> Dict[str, np.ndarray]:
    """
    Tải tất cả khuôn mặt đã đăng ký từ thư mục dữ liệu.
    Hỗ trợ TƯƠNG THÍCH NGƯỢC: tự động phát hiện file .npy thô cũ chưa mã hóa, 
    nạp vào RAM, tự động mã hóa và ghi đè lại file để bảo mật vĩnh viễn.
    """
    p_dir = Path(faces_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings = {}
    key = load_key()
    
    # Quét cả .npy và .enc đề phòng người dùng đổi đuôi thủ công
    for path in list(p_dir.glob("*.npy")) + list(p_dir.glob("*.enc")):
        if not path.is_file():
            continue
        
        name = path.stem
        try:
            with open(path, "rb") as f:
                header = f.read(6)
            
            # Magic bytes tiêu chuẩn của file NumPy thô là \x93NUMPY
            is_old_npy = (header == b"\x93NUMPY")
            
            if is_old_npy:
                # Đọc theo cấu trúc file numpy chưa mã hóa ban đầu
                raw_emb = np.load(path)
                norm_emb = raw_emb.astype(np.float32)
                
                # Chuẩn hóa Vector đặc trưng L2 norm
                norm = float(np.linalg.norm(norm_emb))
                if norm > 1e-8:
                    norm_emb = (norm_emb / norm).astype(np.float32)
                
                # Lưu đè lại dưới dạng đã mã hóa Fernet
                encrypted_bytes = encrypt_embedding(norm_emb, key)
                with open(path, "wb") as f:
                    f.write(encrypted_bytes)
                
                embeddings[name] = norm_emb
                print(f"[MIGRATION] Da tu dong di tru va ma hoa file npy cu: {path.name}")
            else:
                # File đã được mã hóa an toàn
                with open(path, "rb") as f:
                    encrypted_data = f.read()
                
                decrypted_emb = decrypt_embedding(encrypted_data, key)
                embeddings[name] = decrypted_emb
        except Exception as e:
            print(f"[ERROR] Loi giai ma sinh trac hoc khuon mat '{name}' tu {path.name}: {e}")
            
    return embeddings


def get_base_username(name: str) -> str:
    """Tách tên người dùng gốc từ tên file template (ví dụ: john_1 -> john)."""
    if name is None:
        return ""
    import re
    # Xóa hậu tố _\d+ ở cuối tên nếu có
    match = re.match(r"^(.+?)_\d+$", name)
    if match:
        return match.group(1)
    return name


def save_face_multi_template(name: str, embedding: np.ndarray, faces_dir="faces") -> int:
    """
    Mã hóa và lưu trữ đặc trưng khuôn mặt hỗ trợ Đa mẫu (Multi-template, tối đa 3 mẫu).
    Quay vòng (round-robin) thay thế mẫu cũ nhất (dựa trên mtime) nếu đã có đủ 3 mẫu.
    Trả về số thứ tự mẫu đã lưu (0, 1 hoặc 2).
    """
    p_dir = Path(faces_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    
    # Tìm các mẫu hiện có của user này (hỗ trợ cả .npy và .enc)
    existing_indices = []
    for idx in range(3):
        npy_path = p_dir / f"{name}_{idx}.npy"
        enc_path = p_dir / f"{name}_{idx}.enc"
        if npy_path.exists() or enc_path.exists():
            existing_indices.append(idx)
            
    # Hỗ trợ di trú file gốc không có hậu tố _ (nếu người dùng đã có file john.npy trước đó)
    legacy_npy = p_dir / f"{name}.npy"
    legacy_enc = p_dir / f"{name}.enc"
    if legacy_npy.exists() or legacy_enc.exists():
        # Đổi tên file legacy thành john_0
        if legacy_npy.exists() and not (p_dir / f"{name}_0.npy").exists():
            try:
                legacy_npy.rename(p_dir / f"{name}_0.npy")
            except Exception:
                pass
        elif legacy_enc.exists() and not (p_dir / f"{name}_0.enc").exists():
            try:
                legacy_enc.rename(p_dir / f"{name}_0.enc")
            except Exception:
                pass
        if 0 not in existing_indices:
            existing_indices.append(0)
            
    # Chọn index để lưu
    if len(existing_indices) < 3:
        # Nếu chưa đủ 3 mẫu, lưu vào index tiếp theo chưa có
        save_idx = 0
        for idx in range(3):
            if idx not in existing_indices:
                save_idx = idx
                break
    else:
        # Nếu đã có đủ 3 mẫu, ta tìm file cũ nhất (mtime nhỏ nhất) để ghi đè
        save_idx = 0
        oldest_time = float("inf")
        for idx in range(3):
            for ext in [".npy", ".enc"]:
                p = p_dir / f"{name}_{idx}{ext}"
                if p.exists():
                    mtime = p.stat().st_mtime
                    if mtime < oldest_time:
                        oldest_time = mtime
                        save_idx = idx
        
    # Lưu mẫu đã mã hóa
    save_face_encrypted(f"{name}_{save_idx}", embedding, faces_dir)
    return save_idx


def check_anti_spoofing(image: np.ndarray, box: FaceBox) -> Tuple[bool, str, float]:
    """
    Phát hiện giả mạo khuôn mặt (Anti-Spoofing) bằng cách kết hợp:
    1. Độ sắc nét (Laplacian Variance): Phát hiện ảnh in mờ hoặc màn hình phản chiếu.
    2. Phân tích Tần số Cao (FFT Moire): Phát hiện vân sọc quét của màn hình điện thoại/máy tính.
    3. Điểm tin cậy tổng hợp (Liveness Texture Score).
    Trả về: (is_real, reason, score)
    """
    if image is None or image.size == 0 or box is None:
        return True, "Genuine", 1.0
        
    top, right, bottom, left = box
    # Crop vùng mặt
    face = image[top:bottom, left:right]
    if face.size == 0:
        return True, "Genuine", 1.0
        
    # 1. Tải ảnh xám và tính Laplacian Variance (Độ sắc nét)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Phân tích tần số FFT (Phát hiện moire màn hình)
    # Resize về kích thước chuẩn 100x100 để đảm bảo phân tích tần số đồng nhất
    resized = cv2.resize(gray, (100, 100))
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    # Năng lượng tần số thấp ở trung tâm (kích thước 50x50)
    low_energy = float(magnitude_spectrum[cy-25:cy+25, cx-25:cx+25].sum())
    total_energy = float(magnitude_spectrum.sum())
    high_energy = total_energy - low_energy
    high_freq_ratio = high_energy / (total_energy + 1e-6)
    
    # 3. Tính điểm tin cậy tổng hợp
    # Ảnh in/màn hình thường có độ sắc nét cực thấp do độ phân giải camera hoặc phản xạ,
    # hoặc có vân moire quá mạnh (tỷ lệ tần số cao đột biến)
    score = 1.0
    is_real = True
    reason = "Genuine"
    
    # Phạt điểm sắc nét thấp (ngưỡng tiêu chuẩn là 90)
    if lap_var < 90.0:
        score -= (90.0 - lap_var) * 0.008
    # Phạt vân Moire tần số cao (ngưỡng tiêu chuẩn là 0.65)
    if high_freq_ratio > 0.65:
        score -= (high_freq_ratio - 0.65) * 4.0
        
    score = max(0.0, min(1.0, score))
    
    if lap_var < 45.0:
        is_real = False
        reason = "Spoof Detected (Low Sharpness)"
    elif high_freq_ratio > 0.72:
        is_real = False
        reason = "Spoof Detected (Screen Moire)"
    elif score < 0.50:
        is_real = False
        reason = "Spoof Detected (Suspicious Texture)"
        
    return is_real, reason, score


def detect_all_faces(image: np.ndarray) -> List[FaceBox]:
    """Phát hiện tất cả khuôn mặt trong ảnh để phục vụ Ghost Mode (chống nhìn trộm)."""
    if image is None or image.size == 0:
        return []

    # Tăng cường ánh sáng yếu (Night Shift AI)
    image = enhance_low_light(image)

    detector = _get_detector()
    if detector is None:
        # Fallback sang dlib
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb, model="hog")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        return []

    boxes = []
    for d in results.detections:
        bbox = d.location_data.relative_bounding_box
        top = int(bbox.ymin * height)
        left = int(bbox.xmin * width)
        bottom = int((bbox.ymin + bbox.height) * height)
        right = int((bbox.xmin + bbox.width) * width)
        boxes.append((max(0, top), min(width, right), min(height, bottom), max(0, left)))
        
    return boxes


def publish_mqtt_event(username: str, status: str):
    """
    Gửi tín hiệu MQTT về Smart Home (Home Assistant) khi mở khóa thành công.
    Tự động bỏ qua nếu không cấu hình MQTT_BROKER trong file .env.
    """
    import os
    import json
    import time
    
    broker = os.getenv("MQTT_BROKER", "").strip()
    if not broker:
        return
        
    port_str = os.getenv("MQTT_PORT", "1883").strip()
    port = int(port_str) if port_str.isdigit() else 1883
    topic = os.getenv("MQTT_TOPIC", "deeplock/biometric").strip()
    
    try:
        import paho.mqtt.client as mqtt
        
        # Tạo MQTT Client với phiên bản giao thức thích hợp (hỗ trợ v5/v3 tương thích ngược)
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1 if hasattr(mqtt, "CallbackAPIVersion") else None)
        client.connect(broker, port, 10)
        
        payload = {
            "device": "DeepLock Pro Biometric Shield",
            "user": username,
            "status": status,
            "timestamp": int(time.time()),
            "action": "open_door" if status == "GRANTED" else "lock_door"
        }
        
        client.publish(topic, json.dumps(payload), qos=1)
        client.disconnect()
        print(f"[MQTT] Da gui su kien sinh trac hoc toi broker '{broker}': {status} cho {username}")
    except Exception as e:
        print(f"[MQTT ERROR] Khong the gui tin hieu smart home: {e}")



