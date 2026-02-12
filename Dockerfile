FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Fix NumPy compatibility
RUN pip install "numpy<2"

# Install PyTorch for CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML stack
RUN pip install \
    transformers \
    accelerate \
    soundfile \
    runpod

WORKDIR /app

COPY handler.py .

CMD ["python3", "handler.py"]


