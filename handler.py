import os
import base64
import torch
import runpod
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ----------------------------
# Lazy-loaded globals
# ----------------------------
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model ONLY when needed
# ----------------------------
def load_model():
    global model, processor

    if model is None or processor is None:
        print("🎵 Loading MusicGen model...")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        )
        model.to(device)
        model.eval()
        print("✅ MusicGen model loaded")

# ----------------------------
# RunPod handler
# ----------------------------
def handler(job):
    """
    Expected input:
    {
      "prompt": "upbeat 90s new jack swing instrumental",
      "duration": 30,
      "preview": false
    }
    """

    try:
        load_model()

        job_input = job.get("input", {})
        prompt = job_input.get("prompt")
        duration = int(job_input.get("duration", 30))
        preview = bool(job_input.get("preview", False))

        if not prompt:
            return {
                "error": "Prompt is required"
            }

        # ----------------------------
        # HARD SAFETY LIMITS (CRITICAL)
        # ----------------------------
        if preview:
            duration = min(duration, 10)
        else:
            duration = min(duration, 60)

        print(f"🎶 Generating audio | duration={duration}s | preview={preview}")

        # ----------------------------
        # Tokenize prompt
        # ----------------------------
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # ----------------------------
        # Generate audio (SAFE SETTINGS)
        # ----------------------------
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=duration * 25
            )

        # ----------------------------
        # Decode audio safely
        # ----------------------------
        audio = processor.batch_decode(
            audio_values,
            sampling_rate=32000
        )[0]

        # ----------------------------
        # Convert audio → base64
        # ----------------------------
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        audio_bytes = audio_tensor.cpu().numpy().tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio": audio_base64,
            "duration": duration,
            "preview": preview,
            "sample_rate": 32000
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return {
            "error": str(e)
        }

# ----------------------------
# Start RunPod worker
# ----------------------------
runpod.serverless.start({
    "handler": handler
})

