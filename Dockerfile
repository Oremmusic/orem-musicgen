FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    runpod \
    audiocraft \
    transformers \
    accelerate \
    soundfile \
    scipy

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
