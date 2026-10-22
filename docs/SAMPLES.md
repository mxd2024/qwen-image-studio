# Generated samples / 生成サンプル / 生成示例

[English](../README.md) · [日本語](../README.ja.md) · [简体中文](../README.zh-CN.md)

These are actual outputs from this application, not promotional mockups. Both images use Turbo INT8 ConvRot, Euler, the official 8-step sigma schedule, CFG 1, KV cache auto/default, and seed **0**. No prompt enhancer or forced VAE tiling was used. Hardware: Windows, RTX PRO 6000 Blackwell 96 GB. Timing includes per-job model loading.

以下はこのアプリの実出力です。中国語：以下图片由本应用实际生成，并非效果示意图。提示词保持原文以便复现。同じseedでも環境や依存バージョンが異なると画素一致は保証しません。不同环境即使使用相同种子，也不保证逐像素一致。

## Text to image / テキスト生成 / 文生图

![Blue ceramic bird](images/turbo-text-to-image.png)

Prompt:

> A small blue ceramic bird on a wooden table, soft morning light, detailed studio photograph.

- Output: **1024×768**, RGBA PNG.
- Elapsed: **18.0 seconds**, including loading.
- No reference images.

## Image edit / 画像編集 / 图像编辑

![Red ceramic bird](images/turbo-image-edit.png)

Prompt:

> Change the blue ceramic bird to a red ceramic bird. Keep everything else the same.

- Reference 1: the blue bird PNG above, **1024×768 RGBA**.
- Output: **1024×768**, RGBA PNG.
- Elapsed: **20.1 seconds**, including loading.
- The output visibly recolors the bird. Texture, contrast, and background also change; this is generative editing, not exact-pixel preservation.
- 鳥の色変更に加え、質感・コントラスト・背景にも変化があります。無変更部分の画素を固定する機能ではありません。
- 鸟的颜色发生变化，纹理、对比度和背景也有所改变；这不是逐像素保持不变的编辑。

## Reproduce / 再現手順 / 复现步骤

1. Follow the README setup and select **Qwen-Image-2.1 Turbo**.
2. Set width 1024, height 768, seed 0; enter the first prompt and generate.
3. Download the result, then add it as reference 1.
4. Enter the second prompt and generate again. Editing follows reference 1's size.

日本語：Turboを選択し、1024×768・seed 0で1つ目の文を生成します。その結果を参照1に追加し、2つ目の文で編集します。

中文：选择Turbo，以1024×768、种子0运行第一个提示词。将输出添加为参考图1，再使用第二个提示词进行编辑。
