import runpod
import torch
import base64
import io
import soundfile as sf
from audiocraft.models import MusicGen

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(
    duration=30,
    top_k=250,
    temperature=1.0,
    cfg_coef=3.0
)

def handler(job):
    try:
        prompt = job["input"].get("prompt", "hip hop instrumental")
        duration = int(job["input"].get("duration", 30))

        model.set_generation_params(duration=duration)

        wav = model.generate([prompt])

        buffer = io.BytesIO()
        sf.write(buffer, wav[0].cpu().numpy().T, 32000, format="WAV")
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "audio": audio_base64,
            "sample_rate": 32000,
            "format": "wav"
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
