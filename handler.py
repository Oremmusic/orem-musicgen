import base64
import io
import torch
import soundfile as sf
import runpod
from audiocraft.models import MusicGen

# ---- BOOT LOG (VERY IMPORTANT) ----
print("🚀 Handler starting...")
print("🔥 Torch version:", torch.__version__)
print("🎵 Loading MusicGen model...")

# ---- LOAD MODEL ONCE ----
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=5)

print("✅ MusicGen model loaded and ready")

# ---- HANDLER FUNCTION ----
def handler(event):
    try:
        inputs = event.get("input", {})
        prompt = inputs.get("prompt", "").strip()
        duration = int(inputs.get("duration", 5))

        if not prompt:
            return {"error": "Prompt is required"}

        # Safety clamp
        duration = max(1, min(duration, 15))
        model.set_generation_params(duration=duration)

        print(f"🎼 Generating audio | {duration}s | prompt='{prompt}'")

        with torch.no_grad():
            wav = model.generate([prompt])

        audio = wav[0].cpu().numpy()

        buffer = io.BytesIO()
        sf.write(buffer, audio, model.sample_rate, format="WAV")
        buffer.seek(0)

        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print("✅ Audio generation complete")

        return {
            "status": "success",
            "audio_base64": audio_b64,
            "sample_rate": model.sample_rate,
            "duration": duration
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

# ---- START SERVERLESS ----
runpod.serverless.start({
    "handler": handler
})
