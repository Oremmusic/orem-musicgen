FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

# ---- INSTALL TORCH FIRST (CRITICAL) ----
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchaudio==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- CORE ML STACK ----
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    soundfile

# ---- AUDIOCRAFT LAST (ISOLATED) ----
RUN pip install --no-cache-dir audiocraft

# Copy app
COPY . .

CMD ["python3", "handler.py"]
