import base64
import torch
import runpod
import io
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# =========================================
# GLOBALS
# =========================================
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================
# LOAD MODEL (ULTRA MODE)
# =========================================
def load_model():
    global model, processor
    if model is None:
        print("🔥 Loading Mero ULTRA (musicgen-large)...")

        processor = AutoProcessor.from_pretrained(
            "facebook/musicgen-large"
        )

        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-large"
        ).to(device)

        model.eval()

# =========================================
# HANDLER
# =========================================
def handler(job):
    try:
        load_model()

        job_input = job["input"]
        prompt = job_input.get("prompt", "High quality instrumental")
        duration = int(job_input.get("duration", 30))

        print(f"🎵 Generating ULTRA beat | {duration}s")

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # =========================================
        # ULTRA GENERATION SETTINGS
        # =========================================
        audio_tokens = model.generate(
            **inputs,
            max_new_tokens=duration * 50,  # increased from 40
            do_sample=True,
            temperature=1.0,
            top_k=250,
            top_p=0.95,
        )

        audio = processor.batch_decode(
            audio_tokens,
            sampling_rate=32000
        )[0]

        # Convert to float32 numpy
        audio_np = torch.tensor(audio, dtype=torch.float32).cpu().numpy()

        # Write WAV to buffer (Stereo ready)
        wav_buffer = io.BytesIO()
        sf.write(
            wav_buffer,
            audio_np,
            samplerate=32000,
            format="WAV",
            subtype="PCM_16"
        )

        wav_bytes = wav_buffer.getvalue()
        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        print("✅ ULTRA Beat Generated")

        return {
            "audio": audio_base64,
            "sample_rate": 32000,
            "format": "wav",
            "mode": "ultra",
            "duration": duration
        }

    except Exception as e:
        print("❌ Mero Ultra Error:", str(e))
        return {"error": str(e)}

# =========================================
# START RUNPOD
# =========================================
runpod.serverless.start({
    "handler": handler
})
