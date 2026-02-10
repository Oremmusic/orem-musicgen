FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 12.1 (stable)
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Core ML stack (NO av, NO audiocraft)
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    soundfile==0.12.1 \
    runpod==1.8.1

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
