from fastapi import FastAPI
from pydantic import BaseModel
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import uuid
import os
import base64

app = FastAPI()

print("Loading MusicGen model...")
model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=5)
print("MusicGen model loaded.")

class BeatRequest(BaseModel):
    prompt: str

@app.get("/")
def health():
    return {"status": "OREM MusicGen server running"}

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.post("/generate")
def generate(req: BeatRequest):
    with torch.no_grad():
        wav = model.generate([req.prompt])[0]

    filename = f"beat_{uuid.uuid4().hex}.wav"
    path = f"/workspace/{filename}"

    audio_write(
        path,
        wav.cpu(),
        model.sample_rate,
        strategy="loudness"
    )

    with open(path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    os.remove(path)

    return {
        "status": "success",
        "audio": audio_base64,
        "filename": filename
    }
