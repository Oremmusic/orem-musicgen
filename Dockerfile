# ================================
# CUDA base (stable for RunPod)
# ================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ================================
# System dependencies (FULL AUDIO)
# ================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswresample-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Python setup
# ================================
RUN python3 -m pip install --upgrade pip

# ----------------
# NumPy pin (CRITICAL)
# ----------------
RUN pip uninstall -y numpy && \
    pip install numpy==1.26.4

# ----------------
# PyTorch (CUDA 12.1)
# ----------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ----------------
# AI + Audio stack (NO audiocraft)
# ----------------
RUN pip install \
    transformers \
    accelerate \
    soundfile \
    runpod \
    av

# ================================
# App directory
# ================================
WORKDIR /app

# ================================
# Copy handler
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
