import runpod
import torch
import base64
import io
import soundfile as sf
from audiocraft.models import MusicGen

# Force CPU fallback off
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load model once at startup
model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(
    duration=30,   # 30 seconds
    top_k=250,
    top_p=0.0,
    temperature=1.0,
    cfg_coef=3.0
)

def handler(job):
    try:
        prompt = job["input"].get("prompt", "hip hop instrumental")
        duration = int(job["input"].get("duration", 30))

        print(f"Generating: {prompt}")
        print(f"Duration: {duration}")

        model.set_generation_params(duration=duration)

        wav = model.generate([prompt])

        wav_buffer = io.BytesIO()
        sf.write(
            wav_buffer,
            wav[0].cpu().numpy().T,
            32000,
            format="WAV"
        )

        audio_bytes = wav_buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio": audio_base64,
            "sample_rate": 32000,
            "format": "wav"
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
