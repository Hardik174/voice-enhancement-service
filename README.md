# Standalone Voice Enhancement API Service

This is a standalone, high-performance API service designed for hosting on GPU (or CPU) instances. It provides in-memory audio denoising (using the causal Demucs speech denoiser) and voice reconstruction/super-resolution (using the Resemble-Enhance vocoder).

It exposes a FastAPI endpoint `/enhance` that takes audio bytes encoded as a Base64 string and returns the cleaned, enhanced 44.1 kHz audio as a Base64 string. 

## Features
- **In-Memory Processing:** Audio is decoded, processed, and re-encoded entirely in-memory using PyTorch tensors and `io.BytesIO`. No disk I/O is performed during requests.
- **Patched Resemble-Enhance:** Includes a local, patched version of the `resemble-enhance` library that completely bypasses `deepspeed` compiler requirements, allowing easy installation on Windows, macOS, and Linux without native build tools.
- **GPU Accelerated:** Automatically uses CUDA/GPU if available for fast inference.

---

## 1. Installation

### Set up Virtual Environment
Create a clean Python virtual environment (Python 3.10 to 3.12 recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
Install the patched local resemble-enhance first, then the remaining requirements and the Demucs model library:

```bash
# 1. Install local patched vocoder
pip install -e ./resemble-enhance

# 2. Install main requirements
pip install -r requirements.txt

# 3. Install the archived upstream speech-denoiser
pip install denoiser
```

---

## 2. Running the Server

Start the API service using Uvicorn. The denoiser model weights will automatically download on startup:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

To run with auto-reload (for development only):
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. API Reference

### POST `/enhance`

Enhance and reconstruct the voices in a Base64-encoded audio file.

**Request Body (JSON):**
```json
{
  "audio_base64": "...", // Required: Base64 string of input WAV, MP3, or FLAC bytes
  "dry": 0.01,           // Optional: dry-mix ratio of input noise (0.0 to 1.0, default: 0.01)
  "resemble": true,      // Optional: run resemble-enhance vocoder stage (default: true)
  "nfe": 64,             // Optional: resemble-enhance ODE solver steps (1 to 128, default: 64)
  "solver": "midpoint",  // Optional: resemble-enhance solver: "rk4", "midpoint", "euler" (default: "midpoint")
  "lambd": 0.9           // Optional: blend ratio between original vs reconstructed audio (0.0 to 1.0, default: 0.9)
}
```

**Response Body (JSON):**
```json
{
  "enhanced_audio_base64": "...", // Base64 string of the output WAV bytes
  "sample_rate": 44100            // Output sample rate (44100 Hz if enhanced, 16000 Hz if denoiser-only)
}
```

---

## 4. Verification

Use the provided `test_client.py` script to test your running API endpoint with local audio files.

```bash
# Send an audio file to the API, get it enhanced, and save it locally as enhanced_output.wav
python test_client.py --in path/to/input.mp3 --out path/to/output.wav --url http://127.0.0.1:8000
```
