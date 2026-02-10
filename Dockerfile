FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python setup
# -----------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# 🔒 HARD PIN: NumPy < 2 (CRITICAL)
RUN pip install "numpy<2"

# -----------------------------
# PyTorch (CUDA 12.1 OFFICIAL)
# -----------------------------
RUN pip install \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# Core runtime deps (NO audiocraft)
# -----------------------------
RUN pip install \
    transformers \
    accelerate \
    soundfile \
    runpod

# -----------------------------
# App
# -----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
