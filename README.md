# 🔐 DeepLock Pro - Advanced Biometric Face Authentication

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&style=flat-square)](https://streamlit.io/)
[![Cryptography](https://img.shields.io/badge/Biometrics-AES--128--Fernet-success?logo=keybase&style=flat-square)](https://cryptography.io/)
[![Liveness](https://img.shields.io/badge/Liveness-Anti--Spoofing-orange?style=flat-square)](#)
[![Low Light](https://img.shields.io/badge/Night%20Shift-CLAHE%20AI-purple?style=flat-square)](#)

**Hệ thống mở khóa khuôn mặt thời gian thực tích hợp Chống giả mạo, Mã hóa sinh trắc học và Trợ lý giọng nói Việt**
</div>

---

**DeepLock Pro** (tên gốc: *FaceUnlock*) là hệ thống mở khóa bằng nhận diện khuôn mặt dùng camera điện thoại qua IVcam/IP Camera hoặc camera ảo USB.
He thong dung model pre-trained cua `face_recognition` de trich xuat embedding 128 chieu, khong train lai model.


## Cau truc

```text
FaceUnlock/
+-- faces/
+-- main.py
+-- face_utils.py
+-- requirements.txt
```

## Cai dat

Neu ban da cai moi truong truoc do va dang chay duoc `python main.py`, khong can cai lai.

Neu tao moi moi truong:

```powershell
cd FaceUnlock
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Chay voi IVcam USB camera ao

Neu ban cam dien thoai qua USB va IVcam/DroidCam tao camera ao tren Windows, thuong chi can dung camera index `1`:

```powershell
python main.py --camera 1
```

Neu khong len hinh, thu cac index khac:

```powershell
python main.py --camera 0
python main.py --camera 2
```

## Chay voi IVcam/IP Camera qua Wi-Fi

1. Cai IVcam hoac IP Webcam tren dien thoai.
2. Ket noi dien thoai va may tinh vao cung mang Wi-Fi.
3. Mo app camera tren dien thoai, lay IP stream, thuong co dang:

```text
http://<IP_DIEN_THOAI>:8080/video
```

Chay va nhap IP khi duoc hoi:

```powershell
python main.py
```

Tuy chinh:

```powershell
python main.py --ip 192.168.1.5
python main.py --url http://192.168.1.5:8080/video
python main.py --camera 1
python main.py --threshold 0.5
```

- `--camera`: camera index tren may tinh, dung cho IVcam USB camera ao.
- `--ip`: IP dien thoai, code tu ghep thanh `http://IP:8080/video`.
- `--url`: URL stream day du.
- `--port`: port stream, mac dinh `8080`.
- `--threshold`: nguong mo khoa cho full-face.

## Phim tat

- `r`: dang ky khuon mat moi, nen thao khau trang.
- `q`: thoat.

Khi dang ky, chuong trinh hoi ten trong terminal va luu embedding vao:

```text
faces/<ten_nguoi_dung>.npy
```

Neu file da ton tai, chuong trinh se hoi truoc khi ghi de.

## Co che xac thuc & Bao mat (Cap nhat Phase 1)

1. **Ma hoa Sinh trac hoc (AES Fernet 128-bit) 🔐**:
   - Cac file dac trung khuon mat `.npy` trong thu muc `faces/` bay gio khong con la du lieu tho. Chung da duoc ma hoa bang thu vien Cryptography su dung thuan toan Fernet.
   - Khoa ma hoa `ENCRYPTION_KEY` duoc luu an toan trong file `.env` o thu muc goc.
   - **MIGRATION TU DONG**: He thong hoan toan tuong thich nguoc. Khi ban khoi chay phien ban moi nay lan dau tien, he thong se tu dong quet cac file `.npy` cu chua ma hoa, nap vao RAM, chuan hoa vector, tien hanh ma hoa bao mat va ghi de an toan lai len dia. Ban khong can lam thu cong bat ky buoc nao!

2. **Tang cuong anh sang yeu (Night Shift AI - CLAHE) 🌙**:
   - He thong tu dong phat hien neu do sang moi truong trung binh cua camera thap hon nguong an toan (75.0).
   - Khi do, bo loc CLAHE se duoc ap dung truc tiep len kenh Lightness (he LAB) de tang do tuong phan va do chi tiet cua khuon mat truoc khi dua vao cac pipeline AI (phat hien va nhan dien). Nhan dien nhay gap 2 lan vao ban dem!

3. **Giong noi tro ly Viet TTS (gTTS + Web Audio) 🗣️**:
   - Khi xac thuc thanh cong hoac phat hien lua dao, tro ly tieng Viet se tu dong phat am thanh thong bao truc tiep tren trinh duyet nguoi dung (vi du: *"Quyen truy cap duoc phe duyet. Chao mung [Ten]!"*).
   - Am thanh duoc tao va phat hoan toan tren RAM (Memory Stream) bang `gTTS` va HTML5 Audio tag an, dam bao khong tao bat ky file rac nao tren o cung, chay da nen tang khong phu thuoc phan cung loa may chu.
   - Sidebar streamlit co toggle: "Enable Voice Guidance 🗣️" de bat/tat tieng cuc ky ton trong trai nghiem nguoi dung.

## Huong dan chay Kiem thu tu dong

De dam bao moi thu luon on dinh truoc khi trien khai thuc te, ban co the chay script kiem thu doc lap (ASCII-safe):

```powershell
python test_encryption.py
```

Script nay se tu dong kiem tra:
- Khoi tao khoa Fernet va ghi file `.env`.
- Tinh toan ma hoa/giai ma trung thuc 100% (sai so < 1e-6).
- Gia lap file `.npy` cu va kiem thu di tru du lieu ma hoa tu dong tren dia.
- Gia lap anh toi va kiem thu bo loc CLAHE hoat dong khong loi.

## Ghi chu

- Neu deo khau trang, chuong trinh crop upper-face bang landmarks roi moi trich embedding.
- Liveness dung Eye Aspect Ratio qua 15 frame lien tiep kem theo check huong dau Head Pose.
- Neu MediaPipe ban hien tai khong co API `mp.solutions`, code fallback sang landmarks cua `face_recognition` de tranh crash tren may da cai truoc do.
