import os
import torch
import runpod
import base64
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model once at startup
try:
    model = MusicGen.get_pretrained("facebook/musicgen-medium", device=device)
    print("MusicGen model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def handler(job):
    global model

    if model is None:
        return {"error": "Model failed to load."}

    job_input = job["input"]
    prompt = job_input.get("prompt", "Upbeat electronic music")
    duration = job_input.get("duration", 10)

    temperature = job_input.get("temperature", 1.0)
    top_k = job_input.get("top_k", 250)
    top_p = job_input.get("top_p", 0.95)
    cfg_coef = job_input.get("classifier_free_guidance", 3.0)

    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
    )

    print(f"Generating music for: {prompt}")

    try:
        # ❌ NO generator argument
        wav = model.generate([prompt], progress=True)

        file_path = f"/tmp/{job['id']}"
        audio_write(
            file_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness",
        )

        with open(f"{file_path}.wav", "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": f"data:audio/wav;base64,{audio_base64}",
            "prompt": prompt,
            "duration": duration,
        }

    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
