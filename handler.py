
import io
import base64
import traceback
import runpod
import torch
import soundfile as sf

from audiocraft.models import MusicGen


# Load model ONCE per worker (important for stability)
MODEL = MusicGen.get_pretrained("facebook/musicgen-small")
MODEL.set_generation_params(duration=10)


def handler(job):
    try:
        # -------------------------
        # Read input
        # -------------------------
        input_data = job.get("input", {})
        prompt = input_data.get("prompt", "")
        duration = int(input_data.get("duration", 10))

        if not prompt:
            return {"error": "Missing prompt"}

        MODEL.set_generation_params(duration=duration)

        # -------------------------
        # Generate audio
        # -------------------------
        with torch.no_grad():
            audio = MODEL.generate([prompt])[0].cpu().numpy()

        # -------------------------
        # Convert to WAV in memory
        # -------------------------
        wav_buffer = io.BytesIO()
        sf.write(
            wav_buffer,
            audio,
            samplerate=32000,
            format="WAV",
            subtype="PCM_16"
        )

        wav_bytes = wav_buffer.getvalue()

        # -------------------------
        # BASE64 encode (REQUIRED)
        # -------------------------
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        # -------------------------
        # Return JSON-safe response
        # -------------------------
        return {
            "audio_base64": audio_b64,
            "sample_rate": 32000,
            "format": "wav"
        }

    except Exception as e:
        print("❌ Handler error:")
        traceback.print_exc()
        return {
            "error": str(e)
        }


# Start RunPod worker
runpod.serverless.start({
    "handler": handler
})
