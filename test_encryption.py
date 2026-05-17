import os
import shutil
import numpy as np
import cv2
import time
from pathlib import Path
from face_utils import (
    load_key,
    encrypt_embedding,
    decrypt_embedding,
    save_face_encrypted,
    load_known_faces_encrypted,
    enhance_low_light,
    get_base_username,
    save_face_multi_template,
    check_anti_spoofing
)

def run_tests():
    print("="*60)
    print("--- STARTING DEEPLOCK PRO SYSTEM TESTS (PHASE 1 & 2) ---")
    print("="*60)
    
    # --- TEST 1: Khởi tạo khóa bảo mật ---
    print("\n[TEST 1] Kiem tra co che sinh khoa va luu tep .env...")
    try:
        key = load_key()
        assert len(key) > 0, "Khoa rong!"
        assert Path(".env").exists(), "Tep .env khong duoc tao!"
        print("[OK] Thanh cong! Khoa Fernet da duoc tao va luu tru an toan.")
    except Exception as e:
        print(f"[ERROR] That bai Test 1: {e}")
        return

    # --- TEST 2: Mã hóa & Giải mã Sinh trắc học ---
    print("\n[TEST 2] Kiem tra tinh tron ven cua du lieu sau ma hoa/giai ma...")
    try:
        mock_embedding = np.random.rand(128).astype(np.float32)
        encrypted_bytes = encrypt_embedding(mock_embedding, key)
        assert isinstance(encrypted_bytes, bytes) and len(encrypted_bytes) > 0, "Du lieu ma hoa bi loi!"
        decrypted_embedding = decrypt_embedding(encrypted_bytes, key)
        diff = np.abs(mock_embedding - decrypted_embedding)
        max_diff = np.max(diff)
        assert max_diff < 1e-6, f"Sai so qua lon: {max_diff}"
        print(f"[OK] Thanh cong! Sai so khoi phuc cuc nho: {max_diff:.2e} (Dat chuan < 1e-6).")
    except Exception as e:
        print(f"[ERROR] That bai Test 2: {e}")
        return

    # --- TEST 3: Tương thích ngược (Di trú dữ liệu cũ tự động) ---
    print("\n[TEST 3] Kiem tra kha nang tu dong di tru va ma hoa file npy cu...")
    test_faces_dir = Path("test_faces_temp")
    if test_faces_dir.exists():
        shutil.rmtree(test_faces_dir)
    test_faces_dir.mkdir()
    
    try:
        old_face_name = "test_user_legacy"
        old_face_emb = np.random.rand(128).astype(np.float32)
        old_face_emb = old_face_emb / np.linalg.norm(old_face_emb)
        legacy_path = test_faces_dir / f"{old_face_name}.npy"
        np.save(legacy_path, old_face_emb)
        
        with open(legacy_path, "rb") as f:
            header = f.read(6)
        assert header == b"\x93NUMPY", "File npy tho tao bi sai chuan!"
        
        db = load_known_faces_encrypted(faces_dir=test_faces_dir)
        assert old_face_name in db, "Khong nhan dang duoc ten khuon mat cu!"
        diff = np.abs(old_face_emb - db[old_face_name])
        assert np.max(diff) < 1e-6, "Giai ma vector di tru bi sai lech!"
        
        with open(legacy_path, "rb") as f:
            new_header = f.read(6)
        assert new_header != b"\x93NUMPY", "Tep tren dia chua duoc tu dong ma hoa bao mat!"
        print("[OK] Thanh cong! He thong da tu phat hien, nang cap bao mat va di tru tep cu sang ma hoa Fernet khong loi.")
    except Exception as e:
        print(f"[ERROR] That bai Test 3: {e}")
        return
    finally:
        if test_faces_dir.exists():
            shutil.rmtree(test_faces_dir)

    # --- TEST 4: Tăng cường ánh sáng yếu (CLAHE) ---
    print("\n[TEST 4] Kiem tra bo loc tang sang thong minh (CLAHE)...")
    try:
        dark_image = np.zeros((300, 300, 3), dtype=np.uint8)
        enhanced_image = enhance_low_light(dark_image, brightness_threshold=75.0)
        assert enhanced_image is not None
        assert enhanced_image.shape == dark_image.shape
        
        semi_dark_image = np.ones((300, 300, 3), dtype=np.uint8) * 10
        enhanced_semi = enhance_low_light(semi_dark_image, brightness_threshold=75.0)
        avg_orig = np.mean(cv2.cvtColor(semi_dark_image, cv2.COLOR_BGR2GRAY))
        avg_enh = np.mean(cv2.cvtColor(enhanced_semi, cv2.COLOR_BGR2GRAY))
        assert avg_enh > avg_orig, f"Bo loc khong tang cuong anh sang! {avg_enh} <= {avg_orig}"
        print(f"[OK] Thanh cong! Bo loc CLAHE tang cuong anh sang tu {avg_orig:.1f} len {avg_enh:.1f} thanh cong.")
    except Exception as e:
        print(f"[ERROR] That bai Test 4: {e}")
        return

    # --- TEST 5: Quản lý Đa mẫu khuôn mặt (Multi-Template Rollback) ---
    print("\n[TEST 5] Kiem tra quan ly da mau khuon mat (Multi-Template) va Quay vong slot...")
    test_faces_dir = Path("test_faces_temp")
    if test_faces_dir.exists():
        shutil.rmtree(test_faces_dir)
    test_faces_dir.mkdir()
    
    try:
        username = "alex"
        
        # 1. Lưu liên tục 4 mẫu đặc trưng
        for i in range(4):
            emb = np.random.rand(128).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            save_idx = save_face_multi_template(username, emb, faces_dir=test_faces_dir)
            
            # Kiểm tra index trả về
            if i < 3:
                assert save_idx == i, f"Index mau luu bi sai! Cho doi {i}, nhung nhan duoc {save_idx}"
            else:
                # Lần thứ 4 sẽ quay vòng thay thế mẫu cũ nhất (index 0) do file alex_0 được tạo đầu tiên
                assert save_idx == 0, f"Chua quay vong dung! Cho doi thay the mau 0, nhung luu vao {save_idx}"
            
            # Sleep 100ms để mtime phân biệt rõ ràng
            time.sleep(0.1)
            
        # 2. Kiểm tra tổng số tệp tin mẫu sinh ra (luôn tối đa 3 mẫu)
        files = list(test_faces_dir.glob("alex_*"))
        assert len(files) == 3, f"So luong file mau luu tren dia vuot qua 3! Phat hien: {len(files)}"
        
        # 3. Kiểm tra tách tên người dùng gốc
        assert get_base_username("alex_0") == "alex", "Tach base name sai!"
        assert get_base_username("alex_2") == "alex", "Tach base name sai!"
        assert get_base_username("alex") == "alex", "Tach base name legacy sai!"
        
        print("[OK] Thanh cong! Quan ly da mau (Multi-Template) va co che tu dong quay vong 3 slot hoat dong 100% dung thiet ke.")
    except Exception as e:
        print(f"[ERROR] That bai Test 5: {e}")
        return
    finally:
        if test_faces_dir.exists():
            shutil.rmtree(test_faces_dir)

    # --- TEST 6: Chống giả mạo Anti-Spoofing (Laplacian & FFT Moire) ---
    print("\n[TEST 6] Kiem tra he thong chong gia mao (Anti-Spoofing) Liveness Texture...")
    try:
        # 1. Tạo ảnh giả lập khuôn mặt chuẩn (Real face - sắc nét tự nhiên, không moire)
        real_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Vẽ các cấu trúc hình học sắc nét để tăng Laplacian Variance tự nhiên
        cv2.rectangle(real_img, (40, 40), (160, 160), (220, 220, 220), -1)
        cv2.rectangle(real_img, (70, 70), (130, 130), (50, 50, 50), -1)
        # Thêm nhiễu hạt nhẹ
        noise = np.random.normal(0, 4, real_img.shape).astype(np.int16)
        real_img = np.clip(real_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Chạy kiểm thử ảnh thật
        is_real, reason, score = check_anti_spoofing(real_img, (0, 200, 200, 0))
        assert is_real == True, f"Nhan dien sai anh that! Diem: {score:.2f}, Ly do: {reason}"
        
        # 2. Tạo ảnh giả lập Spoof (Bị nhòe mờ do in ấn hoặc phản chiếu)
        spoof_blur_img = cv2.GaussianBlur(real_img, (25, 25), 0)
        is_real_spoof, reason_spoof, score_spoof = check_anti_spoofing(spoof_blur_img, (0, 200, 200, 0))
        assert is_real_spoof == False, "Khong phat hien duoc gia mao lam mo!"
        assert "Low Sharpness" in reason_spoof, f"Ly do phat hien sai! Nhan: {reason_spoof}"
        
        # 3. Tạo ảnh giả lập Moire màn hình (Chứa tần số cao dạng lưới sọc ngang dọc)
        spoof_moire_img = real_img.copy()
        # Tạo lưới Moire sọc đen sậm định kỳ cách nhau 8 pixel để sống sót sau resize
        for i in range(0, 200, 8):
            spoof_moire_img[i:i+3, :, :] = 0
            spoof_moire_img[:, i:i+3, :] = 0
        is_real_moire, reason_moire, score_moire = check_anti_spoofing(spoof_moire_img, (0, 200, 200, 0))
        assert is_real_moire == False, "Khong phat hien duoc gia mao man hinh Moire!"
        assert "Screen Moire" in reason_moire, f"Ly do phat hien sai! Nhan: {reason_moire}"
        
        print(f"[OK] Thanh cong! Khoi chan gia mao bi loai bo vi do sac net ({score_spoof*100:.1f}%) hoac Moire ({score_moire*100:.1f}%).")
    except Exception as e:
        print(f"[ERROR] That bai Test 6: {e}")
        return

    print("\n" + "="*60)
    print("ALL TESTS PASSED SUCCESSFULLY! DEEPLOCK IS SECURED AND OPTIMIZED.")
    print("="*60)

if __name__ == "__main__":
    run_tests()
