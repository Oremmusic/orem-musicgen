FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# 🔥 Install PyTorch w/ CUDA 12.1
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 🔥 Install MusicGen + dependencies
RUN pip3 install \
    audiocraft \
    transformers \
    sentencepiece \
    accelerate \
    huggingface_hub

WORKDIR /app

COPY handler.py .

CMD ["python3", "-u", "handler.py"]


