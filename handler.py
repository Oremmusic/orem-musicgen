import base64
import torch
import runpod
import soundfile as sf
import io
from transformers import MusicgenForConditionalGeneration, AutoProcessor

model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, processor
    if model is None:
        print("🎵 Loading MusicGen model...")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        model.to(device)
        model.eval()
        print("✅ MusicGen model loaded")

def handler(job):
    try:
        load_model()

        job_input = job["input"]
        prompt = job_input.get("prompt", "")
        duration = int(job_input.get("duration", 15))

        if not prompt:
            return {"error": "Prompt is required"}

        print(f"🎶 Generating audio | duration={duration}s")

        inputs = processor(text=[prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=duration * 50
            )

        audio = audio_values[0].cpu().numpy()

        # Proper WAV encoding
        buffer = io.BytesIO()
        sf.write(buffer, audio.T, 32000, format="WAV")
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "audio": audio_base64,
            "sample_rate": 32000,
            "duration": duration
        }

    except Exception as e:
        print("❌ Handler error:", str(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
