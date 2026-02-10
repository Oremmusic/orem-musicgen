FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# System dependencies (CRITICAL)
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------------
# Upgrade pip tooling
# -----------------------------
RUN pip install --upgrade pip setuptools wheel

# -----------------------------
# PyTorch (CUDA 12.1)
# -----------------------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# Core ML stack (NO audiocraft yet)
# -----------------------------
RUN pip install \
    transformers \
    accelerate \
    soundfile \
    runpod

# -----------------------------
# Audiocraft (LAST, after system deps)
# -----------------------------
RUN pip install audiocraft

# -----------------------------
# Copy handler
# -----------------------------
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
