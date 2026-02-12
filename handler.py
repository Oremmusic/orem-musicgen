import runpod
import torch
import base64
import io
import soundfile as sf
from audiocraft.models import MusicGen

print("🎵 Loading MusicGen model...")
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=30)
print("✅ MusicGen model loaded")


def handler(event):
    try:
        input_data = event.get("input", {})

        prompt = input_data.get("prompt", "instrumental beat")
        duration = int(input_data.get("duration", 30))

        print(f"🎶 Generating audio | duration={duration}s")

        model.set_generation_params(duration=duration)

        with torch.no_grad():
            wav = model.generate([prompt])

        # Convert tensor to CPU numpy
        audio = wav[0].cpu().numpy()

        # Ensure shape is correct (mono)
        if len(audio.shape) > 1:
            audio = audio[0]

        # Normalize safely
        max_val = max(abs(audio).max(), 1e-6)
        audio = audio / max_val

        # Write WAV into memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, 32000, format="WAV")
        buffer.seek(0)

        # Base64 encode
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "duration": duration
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
