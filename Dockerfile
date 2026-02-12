FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pin numpy to avoid torch conflict
RUN pip install "numpy<2"

# Install Torch CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core packages (NO av)
RUN pip install \
    runpod \
    transformers \
    accelerate \
    soundfile \
    audiocraft

# Copy app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
