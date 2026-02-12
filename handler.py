import runpod
import torch
import base64
import io
import numpy as np
import soundfile as sf
from audiocraft.models import MusicGen

# Load model once at startup
print("🎵 Loading MusicGen model...")
model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=30)
print("✅ MusicGen model loaded")


def handler(job):
    try:
        job_input = job["input"]
        prompt = job_input.get("prompt", "lo-fi hip hop instrumental")
        duration = int(job_input.get("duration", 30))

        print(f"🎶 Generating audio | duration={duration}s")

        model.set_generation_params(duration=duration)

        wav = model.generate([prompt])

        # wav shape: [batch, channels, samples]
        audio = wav[0].cpu().numpy()

        # Convert from (channels, samples) → (samples, channels)
        audio = np.transpose(audio)

        wav_buffer = io.BytesIO()

        sf.write(
            wav_buffer,
            audio,
            samplerate=32000,
            format="WAV"
        )

        wav_bytes = wav_buffer.getvalue()

        # Base64 encode so JSON can return it
        encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")

        return {
            "audio_base64": encoded_audio,
            "sample_rate": 32000,
            "format": "wav"
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
