# Validation record — 2026-10-22

## Actual GPU and browser execution

Windows, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB. The app was started through `run.ps1`, then operated in a real browser at localhost.

| Check | Observed result |
|---|---|
| Turbo text-to-image, seed 0 | 1024×768 output; 8 steps, CFG 1; 18.0 s including loading |
| Turbo edit, 2 references | Reference 1: 1024×768 RGBA with alpha 128–255; reference 2: 512×384 RGB; 1024×768 output; 20.2 s |
| Turbo clean sample edit | One 1024×768 RGBA reference (unmodified generated PNG); 1024×768 output; 20.1 s |
| Edit output size | Followed reference 1 despite UI canvas fields being set to 832×1248 |
| UI | Turbo selection, fixed steps/CFG, restoring standard values, 2:3=832×1248, preview and PNG download worked |
| Concurrent request | HTTP 409 while another job ran |
| Progress | Loading, conditioning, denoising, step updates through 8/8, decode, completion arrived through SSE |
| Failure handling | Worker failures reached UI/status; Windows CP932 log failure reproduced by a regression test and fixed |
| Reference order | Unequal-size file reads originally reordered references; sequential reading fixed and browser-verified |
| GPU release | Worker processes exited after jobs; total GPU usage returned to roughly 29 GB from unrelated applications |

Native sampling uses the official manual sigma sequence, Euler, CFG 1, `QwenImage21Cache(auto/default)`, 64-channel latent, and `TextEncodeQwenImage21`. It is not the standard model with a reduced step count. No external image API was used. Other applications and the user's separate ComfyUI installations were not stopped or modified.

## Automated/local CPU checks

- 11 backend regression tests passed in the existing app environment and a fresh isolated environment containing only web/test dependencies.
- 2 native node CPU checks passed: exact manual sigma values; two RGB/RGBA references, first-reference dimensions rounded to 32, and 64-channel latent.
- Python compilation passed for app, worker, and portable setup.
- Ubuntu 24.04 under WSL: Bash syntax and Python compilation passed.
- A GitHub Actions matrix runs backend CPU tests on Windows, macOS, and Ubuntu. Its results are visible in the repository Actions tab.

## Limits of this verification

- Standard BF16 inference was not rerun as part of this Turbo implementation. Existing dispatch behavior is covered by regression checks.
- macOS and Ubuntu GPU/model inference were not exercised on physical machines. macOS currently uses CPU fallback, not MPS.
- 4096px, 10-reference inference, low-VRAM GPUs, exact alpha preservation, and exact unchanged-pixel preservation are not established by these tests.
- Editing samples show model changes in contrast/texture as well as the requested color. This app provides generative editing, not deterministic mask composition.
- Model file sizes and pinned runtime revisions were checked; full cryptographic weight hashes were not revalidated.
