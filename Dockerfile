# Sử dụng base image Python 3.10-slim để cân bằng hiệu năng và dung lượng
FROM python:3.10-slim

# Thiết lập chế độ Python không ghi đè file bytecode (.pyc) lên đĩa
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài đặt các gói phụ thuộc hệ thống cần thiết cho CMake, Dlib, OpenCV và MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép và cài đặt các thư viện Python trước để tối ưu hóa Docker Cache
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ thư mục dự án vào container
COPY . .

# Mở cổng mặc định của Streamlit
EXPOSE 8501

# Khởi chạy dashboard trên cổng 8501, lắng nghe tất cả các IP
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
