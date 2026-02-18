# ============================
# OREM MUSICGEN WORKER
# PyTorch 2.4 + CUDA 12.1
# ============================

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install \
    runpod \
    audiocraft \
    transformers \
    accelerate \
    soundfile \
    scipy

# Create app directory
WORKDIR /app

# Copy handler into container
COPY handler.py /app/handler.py

# Start RunPod worker
CMD ["python", "-u", "handler.py"]
