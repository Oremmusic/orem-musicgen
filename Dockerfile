FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
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
# Core ML stack (NO AUDIOCRAFT)
# -----------------------------
RUN pip install --no-cache-dir \
    runpod \
    torch==2.1.* \
    torchaudio==2.1.* \
    torchvision==0.16.* \
    transformers \
    accelerate \
    soundfile

# -----------------------------
# Copy app
# -----------------------------
COPY handler.py /app/handler.py

# -----------------------------
# Start worker
# -----------------------------
CMD ["python3", "/app/handler.py"]
