import os
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import pytest

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

@pytest.fixture
def temp_faces_dir():
    dir_path = Path("test_faces_temp")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir()
    yield dir_path
    if dir_path.exists():
        shutil.rmtree(dir_path)

def test_load_key():
    """Test 1: Fernet encryption key generation and storage in .env."""
    key = load_key()
    assert len(key) > 0, "Fernet encryption key is empty!"
    assert Path(".env").exists(), ".env file was not created!"

def test_biometric_encryption():
    """Test 2: Data integrity before/after biometric vector encryption/decryption."""
    key = load_key()
    mock_embedding = np.random.rand(128).astype(np.float32)
    encrypted_bytes = encrypt_embedding(mock_embedding, key)
    
    assert isinstance(encrypted_bytes, bytes) and len(encrypted_bytes) > 0, "Encryption returned empty bytes!"
    
    decrypted_embedding = decrypt_embedding(encrypted_bytes, key)
    diff = np.abs(mock_embedding - decrypted_embedding)
    max_diff = np.max(diff)
    
    assert max_diff < 1e-6, f"Parity recovery error too large: {max_diff}"

def test_backward_compatibility(temp_faces_dir):
    """Test 3: Backward compatibility - Auto-migration of legacy unencrypted .npy files."""
    key = load_key()
    old_face_name = "test_user_legacy"
    old_face_emb = np.random.rand(128).astype(np.float32)
    old_face_emb = old_face_emb / np.linalg.norm(old_face_emb)
    
    legacy_path = temp_faces_dir / f"{old_face_name}.npy"
    np.save(legacy_path, old_face_emb)
    
    with open(legacy_path, "rb") as f:
        header = f.read(6)
    assert header == b"\x93NUMPY", "Legacy numpy file was not correctly generated!"
    
    # Load and trigger migration
    db = load_known_faces_encrypted(faces_dir=temp_faces_dir)
    assert old_face_name in db, "Failed to load migrated username!"
    
    diff = np.abs(old_face_emb - db[old_face_name])
    assert np.max(diff) < 1e-6, "Decryption mapping error on migrated vector!"
    
    # Verify file is now encrypted on disk
    with open(legacy_path, "rb") as f:
        new_header = f.read(6)
    assert new_header != b"\x93NUMPY", "Legacy file on disk was not automatically encrypted!"

def test_low_light_filter():
    """Test 4: Ambient low-light image enhancement filter (CLAHE)."""
    dark_image = np.zeros((300, 300, 3), dtype=np.uint8)
    enhanced_image = enhance_low_light(dark_image, brightness_threshold=75.0)
    
    assert enhanced_image is not None
    assert enhanced_image.shape == dark_image.shape
    
    semi_dark_image = np.ones((300, 300, 3), dtype=np.uint8) * 10
    enhanced_semi = enhance_low_light(semi_dark_image, brightness_threshold=75.0)
    
    avg_orig = np.mean(cv2.cvtColor(semi_dark_image, cv2.COLOR_BGR2GRAY))
    avg_enh = np.mean(cv2.cvtColor(enhanced_semi, cv2.COLOR_BGR2GRAY))
    
    assert avg_enh > avg_orig, f"CLAHE did not enhance brightness: {avg_enh} <= {avg_orig}"

def test_multi_template_management(temp_faces_dir):
    """Test 5: Multi-template management and round-robin template slot overrides."""
    username = "alex"
    
    # Save 4 templates sequentially
    for i in range(4):
        emb = np.random.rand(128).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        save_idx = save_face_multi_template(username, emb, faces_dir=temp_faces_dir)
        
        if i < 3:
            assert save_idx == i, f"Expected template index {i}, got {save_idx}"
        else:
            # 4th template replaces the oldest (index 0)
            assert save_idx == 0, f"Expected slot replacement at index 0, got {save_idx}"
        
        time.sleep(0.1) # Sleep to guarantee distinct mtime
        
    # Verify files on disk (maximum of 3 templates)
    files = list(temp_faces_dir.glob("alex_*"))
    assert len(files) == 3, f"Expected exactly 3 template files, found {len(files)}"
    
    # Verify base username parsing
    assert get_base_username("alex_0") == "alex"
    assert get_base_username("alex_2") == "alex"
    assert get_base_username("alex") == "alex"

def test_anti_spoofing():
    """Test 6: Anti-spoofing liveness checks using Laplacian variance and FFT Moire."""
    # 1. Simulate real face (sharp structures, no moire)
    real_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    cv2.rectangle(real_img, (40, 40), (160, 160), (220, 220, 220), -1)
    cv2.rectangle(real_img, (70, 70), (130, 130), (50, 50, 50), -1)
    noise = np.random.normal(0, 4, real_img.shape).astype(np.int16)
    real_img = np.clip(real_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    is_real, reason, score = check_anti_spoofing(real_img, (0, 200, 200, 0))
    assert is_real is True, f"Failed to identify real face: {reason} (score={score})"
    
    # 2. Simulate blurred spoof image
    spoof_blur_img = cv2.GaussianBlur(real_img, (25, 25), 0)
    is_real_spoof, reason_spoof, score_spoof = check_anti_spoofing(spoof_blur_img, (0, 200, 200, 0))
    assert is_real_spoof is False, "Failed to detect blurry spoof image!"
    assert "Low Sharpness" in reason_spoof
    
    # 3. Simulate screen Moire pattern
    spoof_moire_img = real_img.copy()
    for i in range(0, 200, 8):
        spoof_moire_img[i:i+3, :, :] = 0
        spoof_moire_img[:, i:i+3, :] = 0
    is_real_moire, reason_moire, score_moire = check_anti_spoofing(spoof_moire_img, (0, 200, 200, 0))
    assert is_real_moire is False, "Failed to detect screen Moire spoof pattern!"
    assert "Screen Moire" in reason_moire
