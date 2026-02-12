FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    pkg-config \
    libsndfile1 \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Fix numpy conflict
RUN pip install "numpy<2"

# Install PyTorch CUDA 12.1
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining libraries
RUN pip install \
    runpod \
    audiocraft \
    transformers \
    accelerate \
    soundfile \
    sentencepiece

# Copy handler
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
