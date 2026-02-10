import base64
import torch
import runpod
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ----------------------------
# Globals (lazy-loaded)
# ----------------------------
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model once per worker
# ----------------------------
def load_model():
    global model, processor
    if model is None:
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
    try:
        load_model()

        job_input = job.get("input", {})
        prompt = job_input.get("prompt")

        # HARD SAFETY LIMIT (critical for stability)
        duration = min(int(job_input.get("duration", 8)), 8)

        if not prompt:
            return { "error": "Prompt is required" }

        print(f"🎶 Generating audio | duration={duration}s")

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            audio_tokens = model.generate(
                **inputs,
                max_new_tokens=duration * 50
            )

        # Decode audio safely
        audio = processor.batch_decode(
            audio_tokens,
            sampling_rate=32000
        )[0]

        # Convert to bytes → Base64 (JSON SAFE)
        audio_bytes = (
            torch.tensor(audio, dtype=torch.float32)
            .cpu()
            .numpy()
            .tobytes()
        )

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "sample_rate": 32000,
            "duration": duration
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return { "error": str(e) }

# ----------------------------
# Start RunPod worker
# ----------------------------
runpod.serverless.start({
    "handler": handler
})
