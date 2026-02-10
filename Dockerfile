# ---------------------------------------
# Base Image (refreshed Ubuntu + CUDA)
# ---------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---------------------------------------
# System Dependencies (SAFE + FIXED)
# ---------------------------------------
RUN apt-get update --fix-missing && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------
# Python Tooling
# ---------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# ---------------------------------------
# Core ML Stack (NO audiocraft, NO av)
# ---------------------------------------
RUN pip install --no-cache-dir \
    "numpy<2" \
    runpod \
    torch==2.1.* \
    torchaudio==2.1.* \
    torchvision==0.16.* \
    transformers \
    accelerate \
    soundfile

# ---------------------------------------
# Copy Worker Code
# ---------------------------------------
COPY handler.py /app/handler.py

# ---------------------------------------
# Start Worker
# ---------------------------------------
CMD ["python3", "handler.py"]


