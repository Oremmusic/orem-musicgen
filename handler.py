import base64
import torch
import runpod
import io
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ----------------------------
# Globals
# ----------------------------
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Lazy load model
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
# RunPod Handler
# ----------------------------
def handler(job):
    """
    Expected input:
    {
        "prompt": "dark trap instrumental",
        "duration": 30
    }
    """

    try:
        load_model()

        job_input = job.get("input", {})
        prompt = job_input.get("prompt")

        # 🔥 DEFAULT TO 30 SECONDS
        duration = int(job_input.get("duration", 30))
        duration = min(duration, 30)  # safety cap

        if not prompt:
            return {"error": "Prompt is required"}

        print(f"🎶 Generating audio | duration={duration}s")

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # 🔥 TOKEN MULTIPLIER FOR TRUE 30 SEC OUTPUT
        TOKENS_PER_SECOND = 45
        max_tokens = duration * TOKENS_PER_SECOND

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=250,
                temperature=1.0
            )

        # Decode waveform
        audio = processor.batch_decode(
            audio_values,
            sampling_rate=32000
        )[0]

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(
            wav_buffer,
            audio,
            32000,
            format="WAV",
            subtype="PCM_16"
        )

        wav_bytes = wav_buffer.getvalue()

        print(f"✅ Generated WAV size: {len(wav_bytes)} bytes")

        return {
            "audio": base64.b64encode(wav_bytes).decode("utf-8"),
            "sample_rate": 32000,
            "format": "wav",
            "duration": duration
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
