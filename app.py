from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import io
import pathlib
import sys
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torchaudio

from model import load_model

# Cross-platform compatibility patch for omegaconf (needed by resemble-enhance)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global model references
denoiser_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global denoiser_model
    print("Loading Demucs denoiser model...")
    try:
        denoiser_model = load_model("dns64", device=device)
        print("Demucs model loaded successfully.")
    except Exception as e:
        print(f"Error loading Demucs model: {e}")
    yield


app = FastAPI(
    title="Voice Enhancement API Service",
    description="Standalone FastAPI endpoint for in-memory speech denoising and super-resolution enhancement.",
    version="1.0.0",
    lifespan=lifespan,
)


class EnhanceRequest(BaseModel):
    audio_base64: str = Field(
        ...,
        description="Base64 encoded string of input audio bytes (WAV, MP3, etc.)",
    )
    dry: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description="Dry-mix ratio of noisy input in the denoised signal",
    )
    run_resemble: bool = Field(
        True,
        alias="resemble",
        description="Whether to run the resemble-enhance vocoder stage",
    )
    resemble_nfe: int = Field(
        64,
        ge=1,
        le=128,
        alias="nfe",
        description="Number of diffusion steps for resemble-enhance",
    )
    resemble_solver: Literal["rk4", "midpoint", "euler"] = Field(
        "midpoint", alias="solver", description="ODE solver for resemble-enhance"
    )
    resemble_lambda: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        alias="lambd",
        description="Balances original vs reconstructed acoustics (0.0-1.0)",
    )

    class Config:
        populate_by_name = True


class EnhanceResponse(BaseModel):
    enhanced_audio_base64: str = Field(
        ..., description="Base64 encoded string of enhanced output WAV bytes"
    )
    sample_rate: int = Field(
        ..., description="Sample rate of the enhanced audio (e.g. 44100 Hz)"
    )


def decode_base64_to_tensor(base64_str: str) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
        audio_bytes = base64.b64decode(base64_str)
        buffer = io.BytesIO(audio_bytes)
        data, sr = sf.read(buffer, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        return wav, sr
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode base64 or load audio: {str(e)}",
        )


def encode_tensor_to_base64(wav: torch.Tensor, sr: int) -> str:
    try:
        import soundfile as sf
        buffer = io.BytesIO()
        wav_numpy = wav.detach().float().cpu()
        if wav_numpy.dim() == 2:
            if wav_numpy.shape[0] == 1:
                wav_numpy = wav_numpy.squeeze(0).numpy()
            else:
                wav_numpy = wav_numpy.transpose(0, 1).numpy()
        else:
            wav_numpy = wav_numpy.numpy()
        
        sf.write(buffer, wav_numpy, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode audio tensor: {e}")


@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_endpoint(req: EnhanceRequest):
    if denoiser_model is None:
        raise HTTPException(
            status_code=503, detail="Denoising model is not loaded."
        )

    # 1. Decode base64 to tensor
    wav, sr = decode_base64_to_tensor(req.audio_base64)

    # 2. Resample and normalize to mono 16kHz for Demucs
    wav = wav.to(device)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Demucs expects 16kHz
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # 3. Run causal Demucs Denoiser
    try:
        with torch.inference_mode():
            # Model expects [batch, channels, time] -> e.g. [1, 1, T]
            estimate = denoiser_model(wav[None])[0]  # returns [1, T]

        # Apply dry mix
        denoised = (1.0 - req.dry) * estimate + req.dry * wav
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Denoising model inference failed: {str(e)}"
        )

    output_sr = sr
    # 4. Optional Resemble-Enhance vocoder stage
    if req.run_resemble:
        try:
            from resemble_enhance.enhancer.inference import (
                enhance as resemble_enhance,
            )

            # resemble_enhance expects 1D float32 tensor
            denoised_1d = denoised.reshape(-1)

            # Run enhancement & super-resolution (outputs 44.1kHz 1D tensor)
            enhanced, output_sr = resemble_enhance(
                denoised_1d,
                sr,
                device=device,
                nfe=req.resemble_nfe,
                solver=req.resemble_solver,
                lambd=req.resemble_lambda,
            )
            denoised = enhanced
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Resemble-Enhance vocoder stage failed: {str(e)}",
            )

    # 5. Encode back to base64
    try:
        base64_output = encode_tensor_to_base64(denoised, output_sr)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode output audio to base64: {str(e)}",
        )

    return EnhanceResponse(
        enhanced_audio_base64=base64_output, sample_rate=output_sr
    )
