# Qwen Image Studio

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

A local browser app for **Qwen-Image-2.1** text-to-image generation and reference-based editing. Choose the standard Diffusers model or **Turbo INT8 ConvRot**. Images stay local; no external image API is used.

| Text to image | Image edit: blue → red ceramic |
|---|---|
| ![Blue ceramic bird generated with Turbo](docs/images/turbo-text-to-image.png) | ![Ceramic bird recolored with Turbo](docs/images/turbo-image-edit.png) |

Generated with this app. Prompts and settings: [sample notes](docs/SAMPLES.md).

## Features and platforms

- Up to 10 RGB/RGBA references, selection order preserved.
- Official Turbo 8-step schedule, Euler, CFG 1, and native KV cache.
- Aspect presets including **2:3 (832×1248)**; progress, errors, preview, PNG download.
- Isolated Turbo environment and headless worker. No ComfyUI HTTP server required.
- Standard model loads lazily and is released before Turbo. Turbo exits after each job. Other applications are never stopped.
- Model selection offers standard BF16 and Turbo; GGUF was removed from the UI.

| Platform | Standard backend | Turbo | Verification |
|---|---|---|---|
| Windows + NVIDIA | CUDA | CUDA | Turbo GPU inference and UI tested on RTX PRO 6000 Blackwell 96 GB |
| Ubuntu/Linux + NVIDIA | CUDA / CPU fallback | CUDA | Portable scripts and CPU CI; GPU not tested here |
| macOS | CPU fallback, float32 | Unavailable | CPU CI; model inference not tested here |

**macOS does not currently use MPS/Metal.** CPU inference is very slow and needs substantial RAM; this is not a small-memory Mac inference solution. Testing on a 96 GB GPU does not establish minimum VRAM for other GPUs or resolutions.

## Install

Install **Git, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/)**, and a compatible NVIDIA driver for CUDA.

```sh
git clone https://github.com/mxd2024/qwen-image-studio.git
cd qwen-image-studio
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-web.txt
```

### Turbo: Windows / Linux NVIDIA

Download the three files from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/0217efd1d9c9452e84f6448903d6e21bfa171ffd), or reuse your existing copies:

| Argument | Repository file | Expected bytes |
|---|---|---:|
| `--transformer` | `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors` | 7,256,787,829 |
| `--encoder` | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 |
| `--vae` | `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 |

Replace the model paths below. Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe setup_turbo.py --transformer "D:\models\qwen_image_2.1_turbo_int8_convrot.safetensors" --encoder "D:\models\qwen3vl_8b_int8_convrot.safetensors" --vae "D:\models\qwen_image_2.1_vae_bf16.safetensors"
```

Ubuntu/Linux:

```sh
.venv/bin/python setup_turbo.py --transformer /path/to/qwen_image_2.1_turbo_int8_convrot.safetensors --encoder /path/to/qwen3vl_8b_int8_convrot.safetensors --vae /path/to/qwen_image_2.1_vae_bf16.safetensors
```

Setup **never downloads or copies model weights**. It verifies file sizes, not hashes. Alternatively, place the three files in `models/turbo/` using the filenames above and omit model arguments. Re-running without arguments reuses saved paths.

Setup creates `.venv-turbo`, fetches pinned ComfyUI into `.vendor/ComfyUI`, installs dependencies, and writes local `turbo_config.json`. Use `--comfy-source /path/to/existing/ComfyUI` to clone a local checkout instead. `setup_turbo.ps1` is a Windows wrapper.

Pinned runtime: ComfyUI `1a82e6d01f395b43275c77acd289af1eb7090bb9`, torch `2.11.0+cu130`, torchvision `0.26.0+cu130`, comfy-kitchen `0.2.39`, comfy-aimdo `0.5.6`. Setup uses `requirements-turbo-lock.txt`; Linux additionally resolves platform-specific PyTorch dependencies.

### Standard model: Windows / macOS / Linux

Windows/Linux NVIDIA: install CUDA PyTorch first, then the standard dependencies:

```sh
uv pip install --python .venv torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv -r requirements.txt
```

macOS / CPU: only the second command is needed. The standard model downloads from `Qwen/Qwen-Image-2.1` on first generation. Cache: `models/hf-cache/`. `QWEN_IMAGE_MODEL` may point to a compatible local directory. `OFFLOAD=1` enables CUDA CPU offload; it does not guarantee sufficient VRAM.

## Launch and use

| Platform | Command |
|---|---|
| Windows PowerShell | `.\run.ps1` |
| Windows cmd | `run.bat` |
| macOS / Ubuntu | `bash run.sh` |

Open **http://127.0.0.1:8321**, select a model, enter a prompt, and click the green generate button. Refresh the page after updating the app.

- Turbo fixes steps to **8**, CFG to **1**; returning to standard restores earlier settings. Negative conditioning has no CFG effect at CFG 1.
- T2I uses the requested width/height (256–4096; multiples of 32 for Turbo).
- Adding references enables editing. Output follows **reference 1**, rounded to multiples of 32. Editing ignores the width/height fields.
- RGBA is preserved for the native VAE path; the official vision encoder composites alpha over white. Output alpha and unedited pixels are not guaranteed to match input exactly. No exact-pixel mask editing is provided.
- Turbo reloads each job; elapsed time includes loading. Native `vae.decode` is used without forced tiling.
- Inference runs offline. Initial dependencies, ComfyUI, and standard-model installation need network access. Keep the server on localhost; public-hosting authentication is not provided.

Turbo sigmas: `1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`, Euler, CFG 1, `QwenImage21Cache(auto/default)`.

## Troubleshooting and tests

| Symptom | Action |
|---|---|
| Turbo needs setup | Run setup with the three model paths; inspect `turbo_config.json`. |
| CUDA unavailable / out of memory | Check driver compatibility and other GPU workloads; reduce resolution or reference count. |
| Standard pipeline missing | Install `requirements.txt`, which pins the required Diffusers revision. |
| Port 8321 busy | Close your previous Studio instance, or start uvicorn on another port. |
| macOS Turbo error | Use standard CPU mode; CUDA Turbo is unavailable on macOS. |

```sh
uv pip install --python .venv -r requirements-dev.txt
.venv/bin/python -m unittest test_turbo -v
.venv-turbo/bin/python check_native_turbo.py
```

On Windows use `.venv\Scripts\python.exe` / `.venv-turbo\Scripts\python.exe`. Native checks require the Turbo environment but run on CPU. GitHub Actions runs CPU regression tests on all three OSes; it does not validate GPU inference. [Actual validation results](docs/VALIDATION.md).

## Repository and upstream terms

Tracked: source, setup, tests, docs, and generated samples. Ignored: environments, downloaded ComfyUI, weights, local configuration, private images, diagnostic output. Legacy GGUF API support remains for compatibility, outside the UI.

Upstream projects/models have their own terms: [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), and the [model card / Qwen Research license](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). This repository distributes neither weights nor vendored upstream code.
