"""Comfy-Org MOSS-Image-SFT INT8 ConvRot, native headless CUDA inference."""
import json
import os
from pathlib import Path
import subprocess
import sys

from moss_worker import event
from turbo_worker import COMFY_REVISION


def run(request, output):
    if request.get("images"):
        raise ValueError("MOSS INT8 does not support reference-image editing")
    config = request["config"]
    revision = subprocess.check_output(["git", "-C", config["comfy"], "rev-parse", "HEAD"], text=True).strip()
    if revision != COMFY_REVISION:
        raise RuntimeError(f"Unexpected ComfyUI revision: {revision}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, config["comfy"])
    sys.argv = [sys.argv[0]]
    import torch
    from PIL import Image
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from comfy_extras.nodes_custom_sampler import (
        BasicScheduler, CFGGuider, KSamplerSelect, RandomNoise, SamplerCustomAdvanced,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS INT8 requires NVIDIA CUDA")
    comfy.utils.set_progress_bar_enabled(False)
    with torch.inference_mode():
        event("status", message="MOSS INT8: DiT・テキストエンコーダ・VAEを読み込み中…")
        model = comfy.sd.load_diffusion_model(config["transformer"], model_options={})
        # Native MOSS defaults to flow shift 5 and multiplier 1.
        model = SkipLayerGuidanceDiTSimple.execute(
            model, start_percent=0.0, end_percent=1.0, double_layers="", single_layers="22")[0]
        clip = comfy.sd.load_clip(ckpt_paths=[config["encoder"]], embedding_directory=[],
                                 clip_type=comfy.sd.CLIPType.MOSS_IMAGE, model_options={})
        state, metadata = comfy.utils.load_torch_file(config["vae"], return_metadata=True)
        vae = comfy.sd.VAE(sd=state, metadata=metadata, dtype=torch.float32)
        vae.throw_exception_if_invalid()
        del state
        event("ready")
        event("status", message="MOSS INT8: プロンプトを処理中…")
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(request["prompt"]))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(request.get("negative_prompt") or ""))
        latent = {"samples": torch.zeros([1, 128, request["height"] // 16, request["width"] // 16],
                                         device=comfy.model_management.intermediate_device())}
        guider = CFGGuider.execute(model, positive, negative, cfg=request["cfg"])[0]
        sampler = KSamplerSelect.execute("euler")[0]
        sigmas = BasicScheduler.execute(model, "normal", request["steps"], 1.0)[0]
        noise = RandomNoise.execute(request["seed"])[0]
        comfy.utils.set_progress_bar_global_hook(
            lambda value, total, preview, **kwargs: event("step", step=value, total=total))
        event("status", message="MOSS INT8: Eulerで生成中…")
        sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
        event("status", message="MOSS INT8: 画像をデコード中…")
        pixels = vae.decode(sampled["samples"])
        image = Image.fromarray(pixels[0, :, :, :3].detach().float().cpu().clamp(0, 1)
                                .mul(255).round().to(torch.uint8).numpy())
        image.save(output)


if __name__ == "__main__":
    request_path, output_path = map(Path, sys.argv[1:])
    run(json.loads(request_path.read_text(encoding="utf-8")), output_path)
