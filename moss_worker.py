"""One local MOSS-Image-SFT job using the pinned official custom pipeline."""
import json
import os
from pathlib import Path
import sys


def event(kind, **values):
    print("STUDIO_EVENT " + json.dumps({"type": kind, **values}, ensure_ascii=False), flush=True)


def run(request, output):
    if request.get("images"):
        raise ValueError("MOSS-Image-SFT supports text/layout generation, not reference-image editing")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import logging as diffusers_logging
    from transformers.utils import logging as transformers_logging

    diffusers_logging.disable_progress_bar()
    transformers_logging.disable_progress_bar()

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-Image-SFT requires an NVIDIA CUDA GPU in this app")
    directory = Path(request["config"]["model"])
    event("status", message="MOSS: 公式モデルを読み込み中…")
    pipe = DiffusionPipeline.from_pretrained(
        str(directory), custom_pipeline=str(directory / "pipeline_moss_image.py"),
        trust_remote_code=True, local_files_only=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    event("ready")
    event("status", message="MOSS: プロンプトを処理して生成中…")

    # The official pipeline iterates progress_bar(timesteps); it has no step callback API.
    def progress(iterable):
        total = len(iterable)
        for index, timestep in enumerate(iterable, 1):
            yield timestep
            event("step", step=index, total=total)
        event("status", message="MOSS: 画像をデコード中…")

    pipe.progress_bar = progress
    with torch.inference_mode():
        image = pipe(
            prompt=request["prompt"], negative_prompt=request.get("negative_prompt") or "",
            width=request["width"], height=request["height"],
            num_inference_steps=request["steps"], guidance_scale=request["cfg"],
            uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(request["seed"]),
        ).images[0]
    image.save(output)


if __name__ == "__main__":
    request_path, output_path = map(Path, sys.argv[1:])
    run(json.loads(request_path.read_text(encoding="utf-8")), output_path)
