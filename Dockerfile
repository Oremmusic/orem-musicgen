FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    ffmpeg \
    pkg-config \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Force NumPy 1.x
RUN pip install "numpy<2"

# Install CUDA 12.1 Torch 2.4
RUN pip install torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install all Python dependencies, ensuring 'av' is installed last to prevent conflicts
RUN pip install \
    runpod \
    transformers==4.41.2 \
    accelerate \
    soundfile \
    scipy \
    audiocraft==1.3.0 \
    av==11.0.0

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
