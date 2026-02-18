# /app/handler.py

import os
import torch
import runpod
import base64
import torchaudio # Often a dependency for audiocraft's audio processing

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Define the device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the MusicGen model once when the worker starts
# The 'device' argument handles moving the model to GPU during loading
try:
    model = MusicGen.get_pretrained('medium', device=device) # Load directly to device
    print(f"MusicGen model ('medium') loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading MusicGen 'medium' model: {e}")
    # Fallback to 'small' model if 'medium' fails or is too big for GPU
    try:
        model = MusicGen.get_pretrained('small', device=device)
        print(f"MusicGen model ('small') loaded successfully on {device} as fallback.")
    except Exception as fallback_e:
        print(f"Fallback to 'small' model failed: {fallback_e}")
        model = None # Ensure model is None if both fail


def handler(job):
    """
    RunPod handler function to generate music using MusicGen.
    """
    global model # Access the pre-loaded model

    if model is None:
        return {"error": "MusicGen model failed to load. Check worker logs."}

    job_input = job['input']
    prompt = job_input.get('prompt', 'Upbeat electronic music')
    duration = job_input.get('duration', 10) # Default to 10 seconds
    
    # Generation parameters (can be passed from frontend)
    temperature = job_input.get('temperature', 1.0)
    top_k = job_input.get('top_k', 250)
    top_p = job_input.get('top_p', 0.95)
    classifier_free_guidance = job_input.get('classifier_free_guidance', 3.0) # Often named cfg_coef
    seed = job_input.get('seed', None) # If None, MusicGen generates a random seed

    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=classifier_free_guidance
    )

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    else:
        generator = None

    print(f"Generating music with prompt: '{prompt}', duration: {duration}s, temp: {temperature}, top_k: {top_k}, top_p: {top_p}, seed: {seed}")
    
    try:
        # Generate music
        wav = model.generate([prompt], progress=True, generator=generator)

        # Save to a temporary file
        audio_path = f"/tmp/{job['id']}.wav"
        # Move audio to CPU for writing
        audio_write(
            stem_name=f"tmp/{job['id']}",
            audio=wav[0].cpu(),
            sample_rate=model.sample_rate,
            format="wav"
        )
        
        # Read the generated WAV file and encode it to base64
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Return the base64 encoded audio
        return {
            "audio_base64": f"data:audio/wav;base64,{audio_base64}",
            "prompt": prompt,
            "duration": duration
        }

    except Exception as e:
        print(f"Error during music generation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Register the handler for RunPod
runpod.serverless.start({"handler": handler})
