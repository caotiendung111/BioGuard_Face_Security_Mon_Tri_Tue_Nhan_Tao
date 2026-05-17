import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import math
import io
import base64
from pathlib import Path
from datetime import datetime
from gtts import gTTS
from dotenv import load_dotenv

from face_utils import (
    detect_face,
    get_embedding,
    get_face_landmarks,
    is_wearing_mask,
    crop_upper_face,
    check_liveness_v2,
    get_head_pose,
    draw_face_mesh,
    load_known_faces_encrypted,
    save_face_encrypted,
    get_base_username,
    save_face_multi_template,
    check_anti_spoofing,
    detect_all_faces,
    publish_mqtt_event
)
from main import (
    best_match,
    normalize_embedding,
    safe_name,
    DEFAULT_THRESHOLD,
    MASK_THRESHOLD
)

# Tải biến môi trường
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
LOG_FILE = BASE_DIR / "access_log.csv"

# Page config
st.set_page_config(
    page_title="DeepLock AI | Enterprise Authentication",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open(BASE_DIR / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "unlocked_until" not in st.session_state:
    st.session_state.unlocked_until = 0.0
if "last_name" not in st.session_state:
    st.session_state.last_name = "Unknown"
if "history_log" not in st.session_state:
    st.session_state.history_log = []
if "last_voiced_event" not in st.session_state:
    st.session_state.last_voiced_event = None

def speak(text: str):
    """Phát âm thanh từ văn bản tiếng Việt hoàn toàn trong bộ nhớ RAM qua HTML5 ẩn."""
    try:
        tts = gTTS(text=text, lang="vi")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        # Nếu mất mạng hoặc gTTS lỗi, không làm đứng chương trình, chỉ hiển thị toast nhỏ
        st.toast(f"Hệ thống giọng nói bận: {e}", icon="🗣️")

def log_access(name, status, score):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"Timestamp": timestamp, "User": name, "Status": status, "Confidence": score}
    
    # Save to CSV
    df = pd.DataFrame([new_entry])
    if not LOG_FILE.exists():
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    
    st.session_state.history_log.append(new_entry)

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='color: #00f2fe;'>💠 DEEPLOCK PRO</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### ⚙️ System Settings")
        show_mesh = st.toggle("Show AI Face Mesh", value=True)
        security_level = st.select_slider("Security Level", options=["Standard", "High", "Ultra"], value="High")
        voice_enabled = st.toggle("Enable Voice Guidance 🗣️", value=True)
        ghost_mode_enabled = st.toggle("Enable Ghost Mode 👻", value=True)
        
        threshold = 0.45
        if security_level == "High": threshold = 0.55
        if security_level == "Ultra": threshold = 0.65

        st.markdown("---")
        st.markdown("### 👥 User Management")
        registered = load_known_faces_encrypted(FACES_DIR)
        
        # Nhóm các mẫu theo tên người dùng gốc (Multi-Template grouping)
        grouped_users = {}
        for full_name in list(registered.keys()):
            base = get_base_username(full_name)
            grouped_users[base] = grouped_users.get(base, 0) + 1
            
        for base, count in grouped_users.items():
            col1, col2 = st.columns([4, 1])
            col1.text(f"👤 {base} ({count} mẫu)")
            if col2.button("❌", key=f"del_{base}"):
                # Xóa tất cả các tệp mẫu của người dùng này (bao gồm cả file legacy cũ)
                deleted_any = False
                for suffix in ["", "_0", "_1", "_2"]:
                    for ext in [".npy", ".enc"]:
                        p = FACES_DIR / f"{base}{suffix}{ext}"
                        if p.exists():
                            p.unlink()
                            deleted_any = True
                st.toast(f"Deleted Identity: {base}", icon="🗑️")
                st.rerun()
        
        new_user = st.text_input("Enroll Identity", placeholder="Full Name")
        enroll_btn = st.button("Capture & Secure", use_container_width=True)

    # Main UI
    st.markdown("<h1>QUANTUM-PRECISION BIOMETRIC SHIELD</h1>", unsafe_allow_html=True)
    
    col_vid, col_data = st.columns([3, 2])
    
    with col_vid:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown('<div class="scanning-line"></div>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time status bar
        status_bar = st.empty()

    with col_data:
        st.markdown("### 📊 Live Analytics")
        m1, m2 = st.columns(2)
        score_metric = m1.empty()
        mask_metric = m2.empty()
        
        st.markdown("### 🧬 Biometric Telemetry")
        m3, m4, m5 = st.columns(3)
        pose_metric = m3.empty()
        liveness_metric = m4.empty()
        spoof_metric = m5.empty()

        st.markdown("---")
        st.markdown("### 📜 Access History")
        log_placeholder = st.empty()

    # Video Feed Loop
    cap = cv2.VideoCapture(0)
    
    blink_history = []
    frame_count = 0

    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # 1. Detection & Mesh (Run every frame for smooth UI)
            # Optimize: Dung frame thap de xu ly AI, nhung hien thi frame cao
            process_frame = cv2.resize(frame, (320, 240))
            box_small = detect_face(process_frame)
            
            box = None
            if box_small is not None:
                h, w = frame.shape[:2]
                bt, br, bb, bl = box_small
                box = (int(bt * h/240), int(br * w/320), int(bb * h/240), int(bl * w/320))

            
            landmarks = get_face_landmarks(process_frame) if box_small is not None else None

            # Ghost Mode: Nhận diện người lạ nhìn trộm (Shoulder Surfing)
            intruder_detected = False
            all_boxes = []
            if box_small is not None:
                h, w = frame.shape[:2]
                if ghost_mode_enabled:
                    all_boxes_small = detect_all_faces(process_frame)
                    bt, br, bb, bl = box_small
                    center_main = ((bt + bb) / 2, (bl + br) / 2)
                    
                    for obox in all_boxes_small:
                        ot, oright, ob, oleft = obox
                        center_o = ((ot + ob) / 2, (oleft + oright) / 2)
                        dist = math.dist(center_main, center_o)
                        
                        # Chuyển đổi tọa độ box về full frame để vẽ
                        obt = int(ot * h/240)
                        obr = int(oright * w/320)
                        obb = int(ob * h/240)
                        obl = int(oleft * w/320)
                        all_boxes.append((obt, obr, obb, obl))
                        
                        if dist > 35.0:  # Khoảng cách tâm lớn chứng tỏ là mặt người khác phía sau
                            intruder_detected = True

            status = "LOCKED"
            curr_name = "Unknown"
            curr_score = 0.0
            mask_status = "Scanning..."
            liveness_status = "Idle"
            pose_text = "Steady"
            spoof_score = 1.0
            spoof_reason = "Genuine"
            color = (0, 0, 255) # Red
            
            is_unlocked = time.time() < st.session_state.unlocked_until
            
            if is_unlocked:
                status = "UNLOCKED"
                curr_name = st.session_state.last_name
                color = (0, 255, 127) # Spring Green
            elif box is not None:
                top, right, bottom, left = box
                
                # 2. Pose Analysis (Fast)
                pitch, yaw, roll = get_head_pose(landmarks)
                pose_text = f"P:{pitch:.1f} Y:{yaw:.1f}"
                
                # 3. Mask Check (Fast)
                wearing_mask = is_wearing_mask(landmarks)
                mask_status = "Detected 😷" if wearing_mask else "Clear ✅"
                
                # 4. Heavy Recognition (Run every 5 frames to reduce lag)
                if frame_count % 5 == 0 and registered:
                    if wearing_mask:
                        face_image, _ = crop_upper_face(frame, landmarks)
                        emb = get_embedding(face_image)
                        target_thresh = MASK_THRESHOLD
                    else:
                        emb = get_embedding(frame, box)
                        target_thresh = threshold
                    
                    if emb is not None:
                        name, score = best_match(emb, registered)
                        st.session_state["tmp_score"] = score
                        st.session_state["tmp_name"] = name
                
                curr_score = st.session_state.get("tmp_score", 0.0)
                curr_name = st.session_state.get("tmp_name", "Unknown")
                
                # 5. Anti-Spoofing Check (Real-time LBP & FFT)
                is_real, spoof_reason, spoof_score = check_anti_spoofing(frame, box)
                
                if intruder_detected:
                    status = "GHOST_ALERT"
                    liveness_status = "INTRUDER DETECTED 👻"
                    color = (0, 69, 255) # Orange-Red
                elif curr_score > threshold:
                    if not is_real:
                        status = "LIVENESS_FAIL"
                        liveness_status = spoof_reason
                    else:
                        # 6. Liveness Check v2 (Fast)
                        blink_history.append(landmarks)
                        if len(blink_history) > 15: blink_history.pop(0)
                        
                        is_live, msg = check_liveness_v2(blink_history)
                        liveness_status = msg
                        
                        if is_live:
                            status = "UNLOCKED"
                            st.session_state.unlocked_until = time.time() + 5.0 # Unlock for 5s
                            st.session_state.last_name = curr_name
                            log_access(curr_name, "GRANTED", curr_score)
                            publish_mqtt_event(curr_name, "GRANTED")
                            blink_history = []
                
                # Visual Overlays
                if show_mesh:
                    draw_face_mesh(display_frame, landmarks)
                
                # Vẽ hộp nhận diện Ghost Mode cho các gương mặt khác
                if ghost_mode_enabled and len(all_boxes) > 1:
                    for obox in all_boxes:
                        ot, oright, ob, oleft = obox
                        # Tránh vẽ đè lên hộp chính của người dùng
                        if obox != (top, right, bottom, left):
                            # Vẽ hộp màu cam cảnh báo người nhìn trộm
                            cv2.rectangle(display_frame, (oleft, ot), (oright, ob), (0, 140, 255), 2)
                            cv2.putText(display_frame, "WARNING: GHOST OBSERVER", (oleft, ot - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                
                # Draw cyber box
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                l = 25
                cv2.line(display_frame, (left, top), (left+l, top), color, 4)
                cv2.line(display_frame, (left, top), (left, top+l), color, 4)
                cv2.line(display_frame, (right, top), (right-l, top), color, 4)
                cv2.line(display_frame, (right, top), (right, top+l), color, 4)
                cv2.line(display_frame, (left, bottom), (left+l, bottom), color, 4)
                cv2.line(display_frame, (left, bottom), (left, bottom-l), color, 4)
                cv2.line(display_frame, (right, bottom), (right-l, bottom), color, 4)
                cv2.line(display_frame, (right, bottom), (right, bottom-l), color, 4)

                # AI Label Overlay
                label = f"SCANNING ID: {curr_name}" if not is_unlocked else f"CLEAR - {curr_name.upper()}"
                cv2.rectangle(display_frame, (left, top - 40), (right, top), color, -1)
                cv2.putText(display_frame, label, (left + 5, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update Frame
            video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
            
            # Update Metrics
            score_metric.metric("Auth Confidence", f"{curr_score:.2f}", delta=f"{curr_score - threshold:.2f}" if curr_score > 0 else None)
            mask_metric.metric("Mask Barrier", mask_status)
            pose_metric.metric("Head Pose", pose_text)
            liveness_metric.metric("Liveness Core", liveness_status)
            spoof_metric.metric("Anti-Spoofing", f"{spoof_score*100:.1f}%", delta="SUSPICIOUS" if spoof_score < 0.6 else "SECURE")
            
            # Update Status Bar
            badge_class = "status-unlocked" if status == "UNLOCKED" else "status-locked"
            status_bar.markdown(f'<div class="status-badge {badge_class}">SYSTEM STATE: {status} | TARGET: {curr_name}</div>', unsafe_allow_html=True)
            
            # --- TRÌNH DIỄN GIỌNG NÓI TTS (PHASE 1 & 2 & 3) ---
            if status == "UNLOCKED":
                event_key = f"unlock_{curr_name}"
                if st.session_state.last_voiced_event != event_key:
                    if voice_enabled:
                        speak(f"Quyền truy cập được phê duyệt. Chào mừng {curr_name}!")
                    st.session_state.last_voiced_event = event_key
            elif status == "LIVENESS_FAIL":
                event_key = f"fail_{liveness_status}"
                if st.session_state.last_voiced_event != event_key:
                    if voice_enabled:
                        if "Spoof" in liveness_status:
                            speak("Cảnh báo. Phát hiện giả mạo khuôn mặt!")
                        else:
                            speak("Truy cập bị từ chối. Vui lòng nháy mắt để xác thực sinh trắc học!")
                    st.session_state.last_voiced_event = event_key
            elif status == "GHOST_ALERT":
                event_key = "ghost_alert"
                if st.session_state.last_voiced_event != event_key:
                    if voice_enabled:
                        speak("Cảnh báo. Phát hiện có người lạ nhìn trộm phía sau lưng!")
                    st.session_state.last_voiced_event = event_key
            elif status == "LOCKED":
                # Reset voiced state khi hệ thống khóa trở lại và không có mặt
                st.session_state.last_voiced_event = None

            # Update History Table
            if LOG_FILE.exists():
                df_log = pd.read_csv(LOG_FILE).tail(5)
                log_placeholder.dataframe(df_log, use_container_width=True, hide_index=True)

            # Enrollment Logic
            if enroll_btn and new_user and box is not None:
                emb = get_embedding(frame, box)
                if emb is not None:
                    norm_emb = normalize_embedding(emb)
                    save_idx = save_face_multi_template(safe_name(new_user), norm_emb, FACES_DIR)
                    st.toast(f"Secured template {save_idx} for: {new_user}", icon="🛡️")
                    time.sleep(1)
                    st.rerun()

            time.sleep(0.01)
    finally:
        cap.release()



if __name__ == "__main__":
    main()
