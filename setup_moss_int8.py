"""Prepare the Comfy-Org MOSS-Image-SFT INT8 ConvRot headless runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from turbo_worker import COMFY_REVISION

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Comfy-Org/MOSS-Image"
MODEL_REVISION = "24147db865fcdc207270f844777587223fed8d6e"
MODEL_FILES = {
    "transformer": "diffusion_models/moss_image_sft_int8_convrot.safetensors",
    "encoder": "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    "vae": "vae/flux2-vae_small.safetensors",
}
MODEL_SIZES = {"transformer": 5413698958, "encoder": 9748121532, "vae": 249521340}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Reuse the local Comfy-Org snapshot/file layout")
    args = parser.parse_args()
    if sys.platform == "darwin":
        parser.error("MOSS INT8 requires NVIDIA CUDA on Windows/Linux in this app")
    comfy = ROOT / ".vendor/ComfyUI"
    if not comfy.exists():
        comfy.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "init", str(comfy)], check=True)
        subprocess.run(["git", "-C", str(comfy), "remote", "add", "origin", "https://github.com/Comfy-Org/ComfyUI.git"], check=True)
        subprocess.run(["git", "-C", str(comfy), "fetch", "--depth=1", "origin", COMFY_REVISION], check=True)
        subprocess.run(["git", "-C", str(comfy), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMFY_REVISION:
        parser.error(f"Expected ComfyUI {COMFY_REVISION}, found {revision}; existing checkout was not changed")
    executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    # Reuse the already pinned native Turbo environment without installing anything into it.
    python = ROOT / ".venv-turbo" / executable
    if not python.exists():
        environment = ROOT / ".venv-moss-int8"
        python = environment / executable
        if not python.exists():
            subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(python), "-r", str(ROOT / "requirements-turbo-lock.txt"),
                        "--extra-index-url", "https://download.pytorch.org/whl/cu130",
                        "--index-strategy", "unsafe-best-match"], check=True)
    model = args.model_dir.absolute() if args.model_dir else None
    if model is None:
        download = (
            "from huggingface_hub import snapshot_download; "
            f"print(snapshot_download({MODEL_ID!r},revision={MODEL_REVISION!r},"
            f"cache_dir={str(ROOT / 'models/hf-cache/hub')!r},allow_patterns={list(MODEL_FILES.values())!r},max_workers=2))"
        )
        model = Path(subprocess.check_output([str(python), "-c", download], text=True).strip())
    config = {"python": str(python), "comfy": str(comfy), "model_id": MODEL_ID, "revision": MODEL_REVISION}
    for name, filename in MODEL_FILES.items():
        path = model / filename
        if not path.is_file() or path.stat().st_size != MODEL_SIZES[name]:
            parser.error(f"Missing or incomplete {name}: {path}")
        config[name] = str(path)
    (ROOT / "moss_int8_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("MOSS INT8 setup complete. Select MOSS-Image-SFT INT8 in the app.")


if __name__ == "__main__":
    main()
