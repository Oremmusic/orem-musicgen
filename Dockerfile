FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip FIRST
RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch (CUDA 12.1 – this is critical)
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Core ML stack (PINNED versions)
RUN pip install \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    soundfile==0.12.1 \
    runpod==1.8.1 \
    av==11.0.0

# ⚠️ DO NOT install audiocraft (it breaks serverless stability)

# App
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
