# MOSS-Image-SFT validation — 2026-10-25

[English](../README.md) · [日本語](../README.ja.md) · [简体中文](../README.zh-CN.md)

## Actual GPU/browser results

Windows; NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB. The app was launched through `run.ps1` and operated from a real browser. Each result used **1024×1024, 50 steps, CFG 3, seed 42**, with no input images. Times include per-job loading and decoding; these are individual observations, not a hardware benchmark or minimum VRAM guarantee.

| Backend | Source revision | Elapsed | Result |
|---|---|---:|---|
| Official Diffusers BF16 | OpenMOSS-Team/MOSS-Image-SFT `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9` | 68.6 s | 1024×1024 PNG, 50/50 step events, browser preview/download |
| Native INT8 ConvRot | Comfy-Org/MOSS-Image `24147db865fcdc207270f844777587223fed8d6e` | 31.0 s | 1024×1024 PNG, 50/50 step events, browser preview/download |

Both worker processes exited after completion. Total GPU usage returned to about 26 GB, belonging to other running applications. Those applications were not stopped. The Qwen virtual environments and pinned ComfyUI checkout were not modified.

## Settings and differences

BF16 uses the official local custom pipeline, BF16 transformer/text encoder, FP32 VAE, CPU random generator, Flow-matching Euler, shift 5, and negative-branch skip layer `[22]`. The official scheduler is reconstructed from its config, preserving its training-grid snapping.

INT8 uses the actual distributed ConvRot DiT and Qwen3.5 text encoder files, native `CLIPType.MOSS_IMAGE`, 128-channel latent, Euler/normal, shift 5, FP32 VAE decode, and negative-branch-only skip layer 22 for the full denoising interval. Runtime: unchanged ComfyUI `1a82e6d01f395b43275c77acd289af1eb7090bb9`, torch 2.11.0+cu130, comfy-kitchen 0.2.39. Its native schedule does not use the official pipeline's training-grid snapping; noise generation and quantization also differ. Equal seeds therefore do not guarantee equal images.

The INT8 DiT is 5,413,698,958 bytes, text encoder 9,748,121,532 bytes, and small VAE 249,521,340 bytes. Setup validates file sizes; full weight hashes were not independently recomputed.

## Sample prompt

> A botanical exhibition poster, cream paper background, detailed green fern illustration, elegant dark green typography, the title reads exactly MOSS IMAGE, smaller subtitle reads LOCAL STUDIO, clean editorial design.

| BF16 | INT8 ConvRot |
|---|---|
| ![BF16 sample](images/moss-sft-bf16.png) | ![INT8 sample](images/moss-sft-int8.png) |

These unretouched outputs render the main requested title/subtitle, but also introduce decorative, misspelled text. Exact typography or pixel identity is not guaranteed.

日本語：BF16・INT8とも実GPUで生成、進捗、保存、終了後のメモリ解放を確認しました。見出し以外に不正確な装飾文字も生成されており、文字の完全一致を保証するものではありません。

中文：BF16和INT8均已实测GPU生成、进度、保存及任务结束后的显存释放。样例中还生成了不准确的装饰文字，不保证所有文字完全准确。

## Contracts and limits

- 17 backend tests cover existing Turbo behavior plus MOSS defaults, explicit settings, reference-image rejection before dispatch, dimensions, worker events/output, seed 0, model release and lock release. Both MOSS variants reject reference images. A CPU import check also confirmed the pinned native MOSS/SLG/scheduler APIs.
- Browser checks confirmed separate sampling defaults, disabled reference upload, retained Qwen references/settings when switching back, fixed Turbo steps, 16-multiple validation, and MOSS result download.
- Official SFT is text/layout-to-image only. No reference-image editing, mask editing, LoRA UI, or prompt enhancer is added.
- CUDA inference was verified on Windows only. Linux GPU inference and macOS model execution are unverified; MOSS is CUDA-only in this app. The CI matrix checks CPU contracts across Windows/macOS/Ubuntu, not GPU execution.
- Qwen's existing worker and weights are unchanged. Its automated regression tests passed; BF16/Qwen GPU inference was not repeated in this MOSS change.

Primary references: [official model](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), [official guide](https://github.com/OpenMOSS/MOSS-Image), [Comfy-Org INT8 files](https://huggingface.co/Comfy-Org/MOSS-Image/tree/24147db865fcdc207270f844777587223fed8d6e), [pinned native implementation](https://github.com/Comfy-Org/ComfyUI/blob/1a82e6d01f395b43275c77acd289af1eb7090bb9/comfy/supported_models.py#L2038).
