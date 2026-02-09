FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV XFORMERS_DISABLE=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install \
    torch==2.1.0 \
    torchaudio==2.1.0

RUN pip3 install audiocraft==1.3.0 --no-deps

RUN pip3 install \
    einops \
    hydra-core \
    hydra-colorlog \
    flashy \
    huggingface_hub \
    transformers \
    sentencepiece \
    num2words \
    tqdm \
    soundfile \
    runpod

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

