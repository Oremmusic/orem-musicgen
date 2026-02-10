# ================================
# Base image with CUDA + cuDNN
# ================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ================================
# System dependencies (CRITICAL)
# ================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Python setup
# ================================
RUN python3 -m pip install --upgrade pip

# ----------------
# Numpy pin (VERY IMPORTANT)
# ----------------
RUN pip uninstall -y numpy && \
    pip install numpy==1.26.4

# ----------------
# PyTorch (CUDA 12.1)
# ----------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ----------------
# AI + Audio stack
# ----------------
RUN pip install \
    transformers \
    accelerate \
    soundfile \
    runpod \
    audiocraft \
    av

# ================================
# App directory
# ================================
WORKDIR /app

# ================================
# Copy source code
# ================================
COPY handler.py /app/handler.py

# ================================
# Environment
# ================================
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# ================================
# Start RunPod worker
# ================================
CMD ["python3", "/app/handler.py"]

