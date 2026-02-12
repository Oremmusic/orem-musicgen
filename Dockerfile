FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Fix numpy conflict
RUN pip install "numpy<2"

# Install CUDA 12.1 compatible torch
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install required libs (NO av)
RUN pip install \
    runpod \
    audiocraft \
    transformers \
    accelerate \
    soundfile \
    sentencepiece

COPY handler.py .

CMD ["python3", "-u", "handler.py"]

