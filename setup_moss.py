"""Install an isolated MOSS-Image-SFT runtime and cache the pinned official model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"
MODEL_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
MODEL_PATTERNS = ["model_index.json", "pipeline_moss_image.py", "scheduler/*", "transformer/*",
                  "text_encoder/*", "tokenizer/*", "vae/*", "LICENSE"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse a complete local official model directory")
    args = parser.parse_args()
    if sys.platform == "darwin":
        parser.error("MOSS requires NVIDIA CUDA on Windows/Linux in this app; macOS is not supported.")
    if args.model and not (args.model / "model_index.json").is_file():
        parser.error("--model must contain a complete official Diffusers model directory")
    environment = ROOT / ".venv-moss"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), "-r", str(ROOT / "requirements-moss.txt"),
                    "--extra-index-url", "https://download.pytorch.org/whl/cu130",
                    "--index-strategy", "unsafe-best-match"], check=True)
    model = args.model.absolute() if args.model else None
    if model is None:
        # Execute with the dedicated environment's Hub client; no global package changes.
        download = (
            "from huggingface_hub import snapshot_download; "
            f"print(snapshot_download({MODEL_ID!r},revision={MODEL_REVISION!r},"
            f"cache_dir={str(ROOT / 'models/hf-cache/hub')!r},allow_patterns={MODEL_PATTERNS!r},max_workers=4))"
        )
        model = Path(subprocess.check_output([str(python), "-c", download], text=True).strip())
    config = {"python": str(python), "model": str(model), "model_id": MODEL_ID,
              "revision": MODEL_REVISION if not args.model else "local"}
    # Config is written only after the full download succeeds.
    (ROOT / "moss_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("MOSS setup complete. Start the app and select MOSS-Image-SFT.")


if __name__ == "__main__":
    main()
