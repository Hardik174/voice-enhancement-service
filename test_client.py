import argparse
import base64
from pathlib import Path
import requests


def test_enhance(input_path: str, output_path: str, url: str):
    input_p = Path(input_path)
    output_p = Path(output_path)

    if not input_p.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    print(f"Reading {input_path}...")
    with open(input_p, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "audio_base64": audio_base64,
        "dry": 0.01,
        "resemble": True,
        "nfe": 64,
        "solver": "midpoint",
        "lambd": 0.9,
    }

    print(f"Sending request to {url}/enhance...")
    response = None
    try:
        response = requests.post(f"{url}/enhance", json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if response is not None:
            print(f"Error Details: {response.text}")
        return

    result = response.json()
    enhanced_base64 = result["enhanced_audio_base64"]
    sample_rate = result["sample_rate"]
    print(f"Received enhanced audio. Sample Rate: {sample_rate} Hz")

    print(f"Saving to {output_path}...")
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, "wb") as f:
        f.write(base64.b64decode(enhanced_base64))
    print("Success! Denoising and enhancement completed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Test client for Voice Enhancement API service."
    )
    p.add_argument(
        "--in", "-i", dest="input", required=True, help="Path to input audio file"
    )
    p.add_argument(
        "--out",
        "-o",
        dest="output",
        default="enhanced_output.wav",
        help="Path to save output enhanced WAV",
    )
    p.add_argument(
        "--url", "-u", default="http://127.0.0.1:8000", help="FastAPI server URL"
    )
    args = p.parse_args()

    test_enhance(args.input, args.output, args.url)
