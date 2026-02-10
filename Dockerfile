FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip safely
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Core ML stack (NO av — this breaks builds)
RUN pip install \
    transformers \
    accelerate \
    soundfile \
    runpod \
    audiocraft

# App
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]

