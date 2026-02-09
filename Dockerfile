FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# 🔥 INSTALL PYTORCH (THIS IS THE MISSING PIECE)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers + RunPod
RUN pip install \
    transformers \
    accelerate \
    scipy \
    runpod

# App
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]


