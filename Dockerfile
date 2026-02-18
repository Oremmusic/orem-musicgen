FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    ffmpeg \
    pkg-config \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install NumPy 1.x (Torch 2.4 + CUDA safe)
RUN pip install "numpy<2"

# Install CUDA 12.1 compatible PyTorch 2.4 FIRST
RUN pip install torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other deps WITHOUT touching torch
RUN pip install \
    runpod \
    transformers \
    accelerate \
    soundfile \
    scipy

# Install audiocraft WITHOUT dependencies (prevents torch downgrade)
RUN pip install audiocraft --no-deps

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
