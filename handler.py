import runpod
import torch
import io
import soundfile as sf
import numpy as np

def handler(job):
    try:
        duration = int(job["input"].get("duration", 10))
        sample_rate = 32000

        # Dummy audio for now (silence)
        audio = np.zeros(sample_rate * duration, dtype=np.float32)

        wav_buffer = io.BytesIO()
        sf.write(
            wav_buffer,
            audio,
            samplerate=sample_rate,
            format="WAV",
            subtype="PCM_16"
        )

        return {
            "audio": wav_buffer.getvalue(),
            "sample_rate": sample_rate,
            "format": "wav"
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return {"error": str(e)}

runpod.serverless.start({
    "handler": handler
})
