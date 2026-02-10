FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Core ML stack (NO av)
RUN pip install \
    runpod \
    torch==2.1.0 \
    torchaudio==2.1.1 \
    torchvision==0.16.0 \
    transformers \
    accelerate \
    soundfile \
    audiocraft

# Copy app
WORKDIR /app
COPY handler.py /app/handler.py

# IMPORTANT: keep process alive
CMD ["python3", "-u", "/app/handler.py"]



