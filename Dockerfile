FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

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
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python tooling
# -----------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Core ML stack (CUDA safe)
# -----------------------------
RUN pip install \
    runpod \
    torch==2.1.* \
    torchaudio==2.1.* \
    torchvision==0.16.* \
    transformers \
    accelerate \
    soundfile \
    audiocraft

# -----------------------------
# App
# -----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
