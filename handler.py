import base64
import torch
import runpod
import io
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ----------------------------
# Globals (lazy loaded)
# ----------------------------
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model once per worker
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
    try:
        load_model()

        job_input = job.get("input", {})
        prompt = job_input.get("prompt")

        # HARD safety cap (critical)
        duration = 8

        if not prompt:
            return {"error": "Prompt is required"}

        print(f"🎶 Generating audio | duration={duration}s")

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=duration * 40
            )

        # Decode audio safely
        audio = processor.batch_decode(
            audio_values,
            sampling_rate=32000
        )[0]

       import io
import soundfile as sf

# Convert to WAV properly (CRITICAL FIX)
audio_np = torch.tensor(audio, dtype=torch.float32).cpu().numpy()

wav_buffer = io.BytesIO()
sf.write(
    wav_buffer,
    audio_np,
    samplerate=32000,
    format="WAV",
    subtype="PCM_16"
)

wav_bytes = wav_buffer.getvalue()



        return {
    "audio": wav_bytes,
    "sample_rate": 32000,
    "format": "wav"
}


    except Exception as e:
        print("❌ Handler error:", str(e))
        return {"error": str(e)}

# ----------------------------
# Start RunPod worker
# ----------------------------
runpod.serverless.start({
    "handler": handler
})


