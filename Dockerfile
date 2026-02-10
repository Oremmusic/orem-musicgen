FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- System deps (REQUIRED for audiocraft / soundfile / av) ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python tooling ----
RUN python3 -m pip install --upgrade pip

# ---- NumPy pin (critical for torch + audiocraft stability) ----
RUN pip uninstall -y numpy && pip install numpy==1.26.4

# ---- PyTorch CUDA 12.1 ----
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Core ML stack ----
RUN pip install \
    transformers \
    accelerate \
    runpod \
    soundfile

# ---- Audiocraft LAST (depends on everything above) ----
RUN pip install audiocraft

# ---- Copy your handler ----
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]


