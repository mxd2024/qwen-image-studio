# Qwen Image Studio

[English](README.md) · **日本語** · [简体中文](README.zh-CN.md)

Qwen-Image-2.1で画像生成・参照画像編集、MOSS-Image-SFTでテキスト／レイアウト生成を行うローカルWebアプリです。Qwen通常版・**Turbo INT8 ConvRot**・**MOSS-Image-SFT**を画面で選べます。画像を外部APIへ送りません。

| テキストから生成 | 同じ画像を青から赤の陶器へ編集 |
|---|---|
| ![Turboで生成した青い陶器の鳥](docs/images/turbo-text-to-image.png) | ![Turboで赤く編集した陶器の鳥](docs/images/turbo-image-edit.png) |

実際のプロンプト・設定は[サンプルの説明](docs/SAMPLES.md)を参照してください。

| MOSS BF16 | MOSS INT8 ConvRot |
|---|---|
| ![MOSS BF16の生成結果](docs/images/moss-sft-bf16.png) | ![MOSS INT8の生成結果](docs/images/moss-sft-int8.png) |

両方とも1024×1024・50ステップ・CFG 3・seed 42の実出力です。[MOSSの検証結果・プロンプト](docs/MOSS_VALIDATION.md)。

## 対応環境

| OS | 通常版 | Turbo | 検証範囲 |
|---|---|---|---|
| Windows + NVIDIA | CUDA | CUDA | RTX PRO 6000 Blackwell 96GBでTurbo生成・編集・UIを実機確認 |
| Ubuntu/Linux + NVIDIA | CUDA / CPU | CUDA | 共通スクリプト・CPUのCI。GPU実機は未検証 |
| macOS | CPU、float32 | 非対応 | CPUのCI。モデル推論の実機は未検証 |

**macOSのMPS/Metalには未対応です。** CPU推論は非常に遅く、大容量メモリが必要です。小容量Macで快適に動く構成ではありません。96GB GPUでの検証結果を、他GPUの最低VRAM要件とみなさないでください。

## 1. アプリを準備

Git、Python 3.12、[uv](https://docs.astral.sh/uv/getting-started/installation/)を用意します。CUDAを使う場合は対応するNVIDIAドライバーも必要です。

```sh
git clone https://github.com/mxd2024/qwen-image-studio.git
cd qwen-image-studio
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-web.txt
```

## 2. Turboを準備（Windows / LinuxのNVIDIA）

MOSSを使う場合は、先に下の「MOSS-Image-SFTを使う場合」を参照してください。Turboの導入は不要です。

[Comfy-Orgのモデル配布ページ](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/0217efd1d9c9452e84f6448903d6e21bfa171ffd)から、次の3ファイルを用意します。既に持っている場合はその実体を再利用できます。

- `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors`
- `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen_image_2.1_vae_bf16.safetensors`

Windows PowerShellの例（モデルの場所を置き換えてください）：

```powershell
.\.venv\Scripts\python.exe setup_turbo.py --transformer "D:\models\qwen_image_2.1_turbo_int8_convrot.safetensors" --encoder "D:\models\qwen3vl_8b_int8_convrot.safetensors" --vae "D:\models\qwen_image_2.1_vae_bf16.safetensors"
```

Ubuntu/Linuxの例：

```sh
.venv/bin/python setup_turbo.py --transformer /path/to/qwen_image_2.1_turbo_int8_convrot.safetensors --encoder /path/to/qwen3vl_8b_int8_convrot.safetensors --vae /path/to/qwen_image_2.1_vae_bf16.safetensors
```

セットアップは重みをダウンロード・コピーしません。サイズを検査し、既存ファイルの場所を`turbo_config.json`へ保存します。ハッシュ検証は行いません。3ファイルを`models/turbo/`へ置けばモデル引数を省略できます。2回目以降は保存済み設定を再利用します。

専用の`.venv-turbo`と固定版`.vendor/ComfyUI`を作り、通常版の環境を変更しません。既存のComfyUI checkoutを読み取り再利用する場合は`--comfy-source /path/to/ComfyUI`を追加します。別のComfyUIサーバーは不要です。

## MOSS-Image-SFTを使う場合（Windows / LinuxのNVIDIA）

「1. アプリを準備」の後に実行します：

```powershell
# Windows
.\.venv\Scripts\python.exe setup_moss.py
```

```sh
# Ubuntu/Linux
.venv/bin/python setup_moss.py
```

専用`.venv-moss`を作成し、[公式モデル](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)約27.3GBを`models/hf-cache/hub`へ取得します。既存のQwen/Turbo環境は変更しません。完全なローカルモデルがある場合は`--model モデルディレクトリ`で再利用できます。設定は公開対象外の`moss_config.json`に保存します。

起動後に**MOSS-Image-SFT**を選びます。公式既定は**50ステップ・CFG 3・Flow-matching Euler・shift 5・負側のlayer [22]スキップ**。Transformer／テキストエンコーダはBF16、VAEはfloat32、乱数生成器はCPUです。サイズは16の倍数を指定し、ステップとCFGは変更できます。モデルごとの設定は切替後も復元されます。

自然文・タグ・JSONレイアウトをプロンプトへ入力できます。公式SFTは**参照画像編集非対応**のため、MOSS選択中は画像追加を無効にし、APIでも画像入力を拒否します。Qwenへ戻すと以前の参照画像を引き続き使えます。macOSでは非対応、LinuxのGPU実機は未検証です。ジョブごとに終了し、他アプリを停止せずGPUメモリを解放します。

重み・公式カスタムコードはrevision `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`に固定し、推論はオフラインです。取得する公式コードは実行されるため、[公式リポジトリ](https://github.com/OpenMOSS/MOSS-Image)を確認のうえ導入してください。MOSSには[配布元のApache-2.0条件](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)が適用されます。

## MOSS INT8 ConvRot版を使う場合

Windowsは`.\.venv\Scripts\python.exe setup_moss_int8.py`、Linuxは`.venv/bin/python setup_moss_int8.py`を実行し、画面で**MOSS-Image-SFT INT8**を選びます。[Comfy-Org配布のINT8版](https://huggingface.co/Comfy-Org/MOSS-Image/tree/24147db865fcdc207270f844777587223fed8d6e)を使用し、FP8やGGUFとは別の形式です。

- DiT: `diffusion_models/moss_image_sft_int8_convrot.safetensors`、約5.41GB。
- テキストエンコーダ: `text_encoders/qwen3.5_9b_int8_convrot.safetensors`、約9.75GB。
- VAE: `vae/flux2-vae_small.safetensors`、約0.25GB。デコードはfloat32。

既存の固定`.vendor/ComfyUI`と`.venv-turbo`があれば、変更せず再利用します。なければ専用`.venv-moss-int8`を作ります。重みの合計約15.4GBを取得し、設定は公開対象外の`moss_int8_config.json`へ保存します。`--model-dir ローカルsnapshot`で同じディレクトリ構成の重みを再利用できます。

既定は50ステップ・CFG 3・Euler・native normalスケジューラ・shift 5・負側layer 22スキップです。公式Diffusersの学習グリッド丸め前に近いスケジュールですが、BF16経路と同じseedでも画素一致は保証しません。こちらもテキスト／レイアウト生成専用、NVIDIA CUDA必須です。別ComfyUIサーバーは起動しません。

## Qwen通常版を使う場合

Windows/LinuxのNVIDIAではCUDA PyTorchを先に入れます：

```sh
uv pip install --python .venv torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv -r requirements.txt
```

macOS / CPUでは2行目だけ実行します。初回生成時に`Qwen/Qwen-Image-2.1`を取得し、`models/hf-cache/`へ保存します。`QWEN_IMAGE_MODEL`で対応するローカルモデルの場所を指定できます。`OFFLOAD=1`はCUDA時のCPUオフロード用です。

## 3. 起動・使い方

| 環境 | コマンド |
|---|---|
| Windows PowerShell | `.\run.ps1` |
| Windows cmd | `run.bat` |
| macOS / Ubuntu | `bash run.sh` |

**http://127.0.0.1:8321**を開き、モデル・プロンプト・サイズを選んで「生成する」を押します。更新後はブラウザを再読み込みしてください。

- Turboは公式8ステップ・Euler・CFG 1・KVキャッシュです。ステップとCFGは固定され、通常版に戻すと以前の値へ戻ります。CFG 1ではネガティブプロンプトによる誘導は働きません。
- 2:3プリセットは**832×1248px**。テキスト生成は指定サイズ、Turboでは256〜4096pxの32の倍数を使います。
- 参照画像を追加すると編集モードになります。最大10枚、RGB/RGBAに対応し、選択順を保ちます。出力サイズは**参照1を32の倍数に丸めた大きさ**です。編集時の幅・高さ欄は無視します。
- RGBAはネイティブVAEへ4チャンネルで渡し、視覚エンコーダ側では公式ノードが白背景に合成します。生成後の透明度や無変更箇所の完全一致を保証するマスク編集ではありません。
- Turboはジョブごとに読み込み、終了時にGPUメモリを解放します。表示時間は読み込み込みです。VAEタイルは強制しません。他アプリを終了させることはありません。
- 推論はオフラインです。初回の依存・ComfyUI・通常モデル取得は通信が必要です。公開ホスト用認証はないためlocalhostで使ってください。

## 困ったとき・検証

- Turboの準備エラー：3つの重みの場所と`turbo_config.json`を確認し、セットアップを再実行してください。
- CUDAエラー／メモリ不足：ドライバーと他アプリのGPU使用量を確認し、解像度や参照数を減らします。
- 通常モデルの読み込みエラー：`requirements.txt`をインストールしてください。必要なDiffusers revisionを固定しています。
- 8321番ポートが使用中：以前のStudioを終了するか、別ポートでuvicornを起動してください。

```powershell
uv pip install --python .venv -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest test_turbo test_moss -v
.\.venv-turbo\Scripts\python.exe check_native_turbo.py
```

macOS/LinuxではPythonのパスを`.venv/bin/python`などに置き換えます。GitHub Actionsは3OSのCPU回帰テストです。GPU推論の合格とは区別しています。[実際の検証結果](docs/VALIDATION.md)・[固定バージョンと詳細](README.md)。

公開対象はソース・手順・テスト・生成サンプルです。重み、仮想環境、取得済みComfyUI、個別設定、私用画像、診断出力は含めません。モデル等には[Qwen Researchライセンスを含む配布元の条件](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)が適用されます。
