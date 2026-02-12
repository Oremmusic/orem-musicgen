import runpod
import torch
import base64
import io
import soundfile as sf
from audiocraft.models import MusicGen

# =========================================================
# DEVICE SETUP
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================================================
# LOAD MODEL (LARGE)
# =========================================================

print("Loading musicgen-large...")
model = MusicGen.get_pretrained("facebook/musicgen-large")
model = model.to(device)

# Default generation params
model.set_generation_params(
    duration=30,
    temperature=1.0,
    top_k=250,
    top_p=0.0
)

print("Model loaded successfully.")

# =========================================================
# HANDLER FUNCTION
# =========================================================

def handler(job):
    try:
        job_input = job.get("input", {})

        prompt = job_input.get("prompt", "")
        duration = int(job_input.get("duration", 30))

        print(f"Generating {duration} seconds...")
        print(f"Prompt: {prompt}")

        # Dynamically override duration
        model.set_generation_params(duration=duration)

        # Generate audio
        wav = model.generate([prompt])

        # Move to CPU + numpy
        audio_np = wav[0].cpu().numpy()

        # Convert to WAV in memory
        buffer = io.BytesIO()
        sf.write(
            buffer,
            audio_np.T,
            32000,
            format="WAV"
        )

        buffer.seek(0)

        # Base64 encode (IMPORTANT for RunPod JSON)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        print("Generation complete.")

        return {
            "audio": audio_base64,
            "sample_rate": 32000,
            "format": "wav",
            "duration": duration,
            "model": "musicgen-large"
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e)
        }

# =========================================================
# START RUNPOD SERVERLESS
# =========================================================

runpod.serverless.start({
    "handler": handler
})
