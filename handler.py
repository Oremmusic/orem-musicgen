import runpod
import torch
from audiocraft.models import MusicGen
import soundfile as sf
import tempfile
import os

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup
print("Loading MusicGen model...")
model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=10)
model = model.to(device)
print("Model loaded successfully.")


def generate_music(job):
    try:
        prompt = job["input"]["prompt"]
        duration = job["input"].get("duration", 10)

        model.set_generation_params(duration=duration)

        print("Generating music for prompt:", prompt)

        wav = model.generate([prompt])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, wav[0].cpu().numpy(), 32000)
            output_path = tmpfile.name

        return {"audio_path": output_path}

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": generate_music})
