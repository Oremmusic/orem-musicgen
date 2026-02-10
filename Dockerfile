# ----------------------------
# Base image (RunPod + CUDA)
# ----------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ----------------------------
# Environment
# ----------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ----------------------------
# System dependencies
# ----------------------------
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

# ----------------------------
# Python tooling
# ----------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# ----------------------------
# CRITICAL: NumPy pin (prevents crashes)
# ----------------------------
RUN pip install "numpy<2"

# ----------------------------
# PyTorch (CUDA 12.1 prebuilt wheels)
# ----------------------------
RUN pip install \
    torch==2.1.* \
    torchvision==0.16.* \
    torchaudio==2.1.* \
    --index-url https://download.pytorch.org/whl/cu121

# ----------------------------
# Core ML stack (NO av)
# ----------------------------
RUN pip install \
    runpod \
    transformers \
    accelerate \
    soundfile \
    audiocraft

# ----------------------------
# App setup
# ----------------------------
WORKDIR /app
COPY . /app

# ----------------------------
# Start RunPod serverless
# ----------------------------
CMD ["python3", "handler.py"]

