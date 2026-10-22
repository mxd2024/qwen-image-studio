"""Create an isolated CUDA Turbo runtime using existing local model files."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from turbo_worker import COMFY_REVISION

ROOT = Path(__file__).resolve().parent
MODEL_SIZES = {"transformer": 7256787829, "encoder": 9350798360, "vae": 675509688}
MODEL_NAMES = {
    "transformer": "qwen_image_2.1_turbo_int8_convrot.safetensors",
    "encoder": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
}


def command(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in MODEL_SIZES:
        parser.add_argument(f"--{key}", type=Path, help=f"Existing local {MODEL_NAMES[key]}")
    parser.add_argument("--comfy-source", type=Path, help="Optional existing ComfyUI git checkout to clone locally")
    args = parser.parse_args()
    if sys.platform == "darwin":
        parser.error("Turbo INT8 requires NVIDIA CUDA on Windows/Linux. Use the standard backend on macOS.")
    for tool in ("git", "uv"):
        if not shutil.which(tool):
            parser.error(f"Install {tool} and add it to PATH first")
    config_path = ROOT / "turbo_config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    config = {}
    for key, expected_size in MODEL_SIZES.items():
        # Keep the .safetensors symlink name: Comfy selects its loader by extension.
        path = (getattr(args, key) or Path(previous.get(key, ROOT / "models" / "turbo" / MODEL_NAMES[key]))).expanduser().absolute()
        if not path.is_file() or path.stat().st_size != expected_size:
            parser.error(f"Missing/incomplete {key} ({expected_size} bytes required): {path}. Specify --{key} PATH")
        config[key] = str(path)
    comfy = ROOT / ".vendor" / "ComfyUI"
    if not comfy.exists():
        comfy.parent.mkdir(exist_ok=True)
        if args.comfy_source:
            command("git", "clone", "--no-hardlinks", "--no-checkout", args.comfy_source.resolve(), comfy)
            command("git", "-C", comfy, "checkout", "--detach", COMFY_REVISION)
        else:
            command("git", "init", comfy)
            command("git", "-C", comfy, "remote", "add", "origin", "https://github.com/Comfy-Org/ComfyUI.git")
            command("git", "-C", comfy, "fetch", "--depth=1", "origin", COMFY_REVISION)
            command("git", "-C", comfy, "checkout", "--detach", "FETCH_HEAD")
    revision = subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMFY_REVISION:
        parser.error(f"Unexpected ComfyUI revision: {revision}. Expected {COMFY_REVISION}")
    environment = ROOT / ".venv-turbo"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        command("uv", "venv", "--python", "3.12", environment)
    command("uv", "pip", "install", "--python", python, "-r", ROOT / "requirements-turbo-lock.txt",
            "--extra-index-url", "https://download.pytorch.org/whl/cu130", "--index-strategy", "unsafe-best-match")
    config.update(python=str(python), comfy=str(comfy))
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Turbo setup complete. Start the app and select Qwen-Image-2.1 Turbo.")


if __name__ == "__main__":
    main()
