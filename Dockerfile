# -----------------------------
# Base image (CUDA 12.1 – stable)
# -----------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# -----------------------------
# Environment settings
# -----------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Upgrade pip tooling
# -----------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Python ML stack
# IMPORTANT:
# - NumPy MUST be < 2
# - Torch installed BEFORE audiocraft
# -----------------------------
RUN pip install \
    "numpy<2" \
    runpod \
    torch \
    torchaudio \
    torchvision \
    transformers \
    accelerate \
    soundfile

# -----------------------------
# Audiocraft (installed LAST)
# -----------------------------
RUN pip install --no-cache-dir audiocraft

# -----------------------------
# App setup
# -----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

# -----------------------------
# Start RunPod worker
# -----------------------------
CMD ["python3", "handler.py"]
