import base64
import io
import torch
import runpod
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# -----------------------------
# Globals (loaded once per worker)
# -----------------------------
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model once per worker
# -----------------------------
def load_model():
    global model, processor

    if model is None or processor is None:
        print("🔄 Loading MusicGen model...")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        ).to(device)
        model.eval()

# -----------------------------
# RunPod handler
# -----------------------------
def handler(job):
    try:
        load_model()

        # -----------------------------
        # Parse input
        # -----------------------------
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "instrumental music")
        duration = int(job_input.get("duration", 30))

        print(f"🎵 Generating audio | Duration: {duration}s | Prompt: {prompt}")

        # -----------------------------
        # Prepare inputs
        # -----------------------------
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # -----------------------------
        # Generate audio tokens
        # -----------------------------
        with torch.no_grad():
            audio_tokens = model.generate(
                **inputs,
                max_new_tokens=duration * 50
            )

        # -----------------------------
        # Decode audio safely
        # -----------------------------
        audio = processor.batch_decode(
            audio_tokens,
            sampling_rate=32000
        )[0]

        audio_tensor = torch.tensor(audio, dtype=torch.float32).cpu().numpy()

        # -----------------------------
        # Write WAV to memory
        # -----------------------------
        wav_buffer = io.BytesIO()
        sf.write(
            wav_buffer,
            audio_tensor,
            samplerate=32000,
            format="WAV",
            subtype="PCM_16"
        )

        wav_bytes = wav_buffer.getvalue()

        # -----------------------------
        # Base64 encode (CRITICAL FIX)
        # -----------------------------
        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        print("✅ Audio generated successfully")

        # -----------------------------
        # Return JSON-safe response
        # -----------------------------
        return {
            "audio": audio_base64,
            "sample_rate": 32000,
            "format": "wav",
            "duration": duration
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return {
            "error": str(e)
        }

# -----------------------------
# Start RunPod serverless worker
# -----------------------------
runpod.serverless.start({
    "handler": handler
})
