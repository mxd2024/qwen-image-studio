"""One local, headless ComfyUI Turbo job. Process exit releases CUDA resources."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import subprocess

SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]
COMFY_REVISION = "1a82e6d01f395b43275c77acd289af1eb7090bb9"


def event(kind, **values):
    print("STUDIO_EVENT " + json.dumps({"type": kind, **values}, ensure_ascii=False), flush=True)


def decode_reference(data):
    from PIL import Image
    image = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[-1])))
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    elif image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    return image


def run(request, output):
    config = request["config"]
    revision = subprocess.check_output(
        ["git", "-C", config["comfy"], "rev-parse", "HEAD"], text=True).strip()
    if revision != COMFY_REVISION:
        raise RuntimeError(f"Unexpected ComfyUI revision: {revision}; run setup_turbo.ps1")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, config["comfy"])
    # Comfy's argument parser must never receive our request/output arguments.
    sys.argv = [sys.argv[0]]
    import numpy as np
    from PIL import Image
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21, QwenImage21Cache
    from comfy_extras.nodes_custom_sampler import (
        CFGGuider, KSamplerSelect, ManualSigmas, RandomNoise, SamplerCustomAdvanced,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Turbo requires a CUDA GPU")
    # Keep stdout as newline-delimited events; interactive tqdm uses carriage returns.
    comfy.utils.set_progress_bar_enabled(False)
    with torch.inference_mode():
        event("status", message="Turbo: 量子化モデル・エンコーダ・VAEを読み込み中…")
        model = comfy.sd.load_diffusion_model(config["transformer"], model_options={})
        model = QwenImage21Cache.execute(model, device="auto", dtype="default")[0]
        clip = comfy.sd.load_clip(ckpt_paths=[config["encoder"]], embedding_directory=[],
                                 clip_type=comfy.sd.CLIPType.QWEN_IMAGE, model_options={})
        vae_state, vae_metadata = comfy.utils.load_torch_file(config["vae"], return_metadata=True)
        vae = comfy.sd.VAE(sd=vae_state, metadata=vae_metadata)
        vae.throw_exception_if_invalid()
        del vae_state
        event("ready")
        event("status", message="Turbo: 参照画像とプロンプトを処理中…")
        images = {f"image_{i}": torch.from_numpy(np.asarray(decode_reference(data)).copy())
                  .float().div(255).unsqueeze(0)
                  for i, data in enumerate(request.get("images", [])[:10], 1)}
        positive, negative, latent = TextEncodeQwenImage21.execute(
            clip=clip, prompt=request["prompt"], negative_prompt=request.get("negative_prompt") or "",
            vae=vae, resolution=0 if images else 1024, images=images).result
        if not images:
            latent = {"samples": torch.zeros(
                [1, 64, request["height"] // 16, request["width"] // 16],
                device=comfy.model_management.intermediate_device())}
        guider = CFGGuider.execute(model, positive, negative, cfg=1.0)[0]
        sampler = KSamplerSelect.execute("euler")[0]
        sigmas = ManualSigmas.execute(",".join(map(str, SIGMAS)))[0]
        noise = RandomNoise.execute(request["seed"])[0]
        comfy.utils.set_progress_bar_global_hook(
            lambda value, total, preview, **kwargs: event("step", step=value, total=total))
        event("status", message="Turbo: 公式8ステップで生成中…")
        sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
        event("status", message="Turbo: 画像をデコード中…")
        pixels = vae.decode(sampled["samples"])
        channels = 4 if pixels.shape[-1] == 4 else 3
        result = Image.fromarray(pixels[0, :, :, :channels].detach().float().cpu()
                                 .clamp(0, 1).mul(255).round().to(torch.uint8).numpy())
        result.save(output)


if __name__ == "__main__":
    request_path, output_path = map(Path, sys.argv[1:])
    run(json.loads(request_path.read_text(encoding="utf-8")), output_path)
