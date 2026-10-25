# Qwen Image Studio

[English](README.md) · [日本語](README.ja.md) · **简体中文**

使用Qwen-Image-2.1在本地生成、编辑图像，并使用MOSS-Image-SFT进行文本／布局生成的浏览器应用。可选择Qwen标准模型、**Turbo INT8 ConvRot**或**MOSS-Image-SFT**。图像在本机处理，不发送到外部图像API。

| 文生图 | 将同一只陶瓷鸟从蓝色改为红色 |
|---|---|
| ![Turbo生成的蓝色陶瓷鸟](docs/images/turbo-text-to-image.png) | ![Turbo编辑后的红色陶瓷鸟](docs/images/turbo-image-edit.png) |

实际提示词和参数见[示例说明](docs/SAMPLES.md)。

| MOSS BF16 | MOSS INT8 ConvRot |
|---|---|
| ![MOSS BF16生成结果](docs/images/moss-sft-bf16.png) | ![MOSS INT8生成结果](docs/images/moss-sft-int8.png) |

两张图均为1024×1024、50步、CFG 3、种子42的实际输出。[MOSS实测结果和提示词](docs/MOSS_VALIDATION.md)。

## 支持的平台

| 系统 | 标准模型 | Turbo | 验证范围 |
|---|---|---|---|
| Windows + NVIDIA | CUDA | CUDA | 已在RTX PRO 6000 Blackwell 96GB上验证Turbo生成、编辑和界面 |
| Ubuntu/Linux + NVIDIA | CUDA / CPU | CUDA | 通用脚本和CPU CI；未验证GPU推理 |
| macOS | CPU、float32 | 不支持 | CPU CI；未实机验证模型推理 |

**目前不使用macOS的MPS/Metal。** CPU推理非常慢，需要大量内存，并非适合小内存Mac的方案。96GB显卡的测试结果不能作为其他显卡或分辨率的最低显存要求。

## 1. 安装应用

准备Git、Python 3.12和[uv](https://docs.astral.sh/uv/getting-started/installation/)。使用CUDA时还需要兼容的NVIDIA驱动。

```sh
git clone https://github.com/mxd2024/qwen-image-studio.git
cd qwen-image-studio
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-web.txt
```

## 2. 安装Turbo环境（Windows / Linux NVIDIA）

从[Comfy-Org模型仓库](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/0217efd1d9c9452e84f6448903d6e21bfa171ffd)准备下列文件，也可以直接复用已有文件：

- `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors`
- `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen_image_2.1_vae_bf16.safetensors`

Windows PowerShell示例，请替换模型路径：

```powershell
.\.venv\Scripts\python.exe setup_turbo.py --transformer "D:\models\qwen_image_2.1_turbo_int8_convrot.safetensors" --encoder "D:\models\qwen3vl_8b_int8_convrot.safetensors" --vae "D:\models\qwen_image_2.1_vae_bf16.safetensors"
```

Ubuntu/Linux示例：

```sh
.venv/bin/python setup_turbo.py --transformer /path/to/qwen_image_2.1_turbo_int8_convrot.safetensors --encoder /path/to/qwen3vl_8b_int8_convrot.safetensors --vae /path/to/qwen_image_2.1_vae_bf16.safetensors
```

安装脚本**不会下载或复制模型权重**，只验证文件大小并将路径写入`turbo_config.json`，不验证哈希。也可将这三个文件放入`models/turbo/`并省略模型参数。再次运行时会复用已保存的路径。

脚本建立独立的`.venv-turbo`和固定版本的`.vendor/ComfyUI`，不会修改标准模型环境。要复用已有ComfyUI源码，可追加`--comfy-source /path/to/ComfyUI`。无需启动ComfyUI HTTP服务器。

## 使用MOSS-Image-SFT（Windows / Linux NVIDIA）

完成「1. 安装应用」后运行：

```powershell
# Windows
.\.venv\Scripts\python.exe setup_moss.py
```

```sh
# Ubuntu/Linux
.venv/bin/python setup_moss.py
```

脚本创建独立的`.venv-moss`，将约27.3GB的[官方模型](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)下载至`models/hf-cache/hub`，不修改Qwen/Turbo环境。已有完整官方模型目录时，可追加`--model /path/to/MOSS-Image-SFT`复用。配置保存在不上传的`moss_config.json`。

启动后选择**MOSS-Image-SFT**。官方默认参数：**50步、CFG 3、Flow-matching Euler、shift 5、负分支跳过第[22]层**。Transformer／文本编码器采用BF16，VAE采用FP32，使用CPU随机数生成器。尺寸须为16的倍数；步数和CFG可调，各模型分别记忆参数。

支持自然语言、标签和JSON布局提示词。官方SFT**不支持参考图编辑**：选择MOSS时禁用图像上传，API也会拒绝图像输入；切回Qwen后保留原有参考图。目前要求NVIDIA CUDA，不支持macOS，Linux GPU推理未实机验证。每次任务结束后释放自身GPU资源，不停止其他程序。

权重和官方自定义代码固定到revision `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`，推理离线运行。安装会下载并执行官方自定义代码，请先查阅[上游项目](https://github.com/OpenMOSS/MOSS-Image)。MOSS遵循[上游Apache-2.0条款](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)。

## 使用MOSS INT8 ConvRot

Windows运行`.\.venv\Scripts\python.exe setup_moss_int8.py`，Linux运行`.venv/bin/python setup_moss_int8.py`，然后选择**MOSS-Image-SFT INT8**。使用[Comfy-Org分发的INT8权重](https://huggingface.co/Comfy-Org/MOSS-Image/tree/24147db865fcdc207270f844777587223fed8d6e)，并非FP8或GGUF：

- DiT：`diffusion_models/moss_image_sft_int8_convrot.safetensors`，约5.41GB。
- 文本编码器：`text_encoders/qwen3.5_9b_int8_convrot.safetensors`，约9.75GB。
- VAE：`vae/flux2-vae_small.safetensors`，约0.25GB，以FP32解码。

如已存在固定版`.vendor/ComfyUI`和`.venv-turbo`，则只读复用，不修改它们；否则创建`.venv-moss-int8`。下载总计约15.4GB，配置写入不上传的`moss_int8_config.json`。可通过`--model-dir /path/to/local/snapshot`复用相同目录结构的本地权重。

默认50步、CFG 3、Euler、原生normal调度、shift 5、负分支跳过第22层。原生调度接近官方Diffusers在训练网格取整前的日程，但两种运行时即使种子相同也不保证像素一致。仅支持文本／布局生成，要求NVIDIA CUDA，无需启动ComfyUI服务器。

## 使用Qwen标准模型

Windows/Linux NVIDIA先安装CUDA版PyTorch，再安装标准模型依赖：

```sh
uv pip install --python .venv torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv -r requirements.txt
```

macOS / CPU只需运行第二行。首次生成时下载`Qwen/Qwen-Image-2.1`，缓存位于`models/hf-cache/`。可用`QWEN_IMAGE_MODEL`指定兼容的本地模型目录。`OFFLOAD=1`用于CUDA模式的CPU卸载。

## 3. 启动与使用

| 环境 | 命令 |
|---|---|
| Windows PowerShell | `.\run.ps1` |
| Windows cmd | `run.bat` |
| macOS / Ubuntu | `bash run.sh` |

打开**http://127.0.0.1:8321**，选择模型，输入提示词和尺寸，点击绿色的「生成する」按钮。更新应用后请刷新页面。当前应用界面为日语。

- Turbo固定为官方8步、Euler、CFG 1和KV缓存。切回标准模型后恢复之前的步数和CFG。CFG 1下负面条件不产生CFG引导效果。
- 2:3预设为**832×1248像素**。文生图使用指定尺寸；Turbo要求256–4096像素且为32的倍数。
- 添加参考图后进入编辑模式，最多10张，支持RGB/RGBA，保持选择顺序。输出尺寸跟随**参考图1，四舍五入到32的倍数**，编辑时忽略宽高输入框。
- RGBA以四通道传给原生VAE；官方节点在视觉编码器端将透明区域合成到白底。输出透明度及未编辑像素不保证与输入完全相同；不提供精确像素蒙版编辑。
- Turbo每次任务重新加载，结束后释放GPU资源，显示时间包含加载。不会强制VAE分块，也不会停止其他应用。
- 推理离线运行。首次安装依赖、ComfyUI及标准模型需要联网。本应用没有公共托管认证，请保持localhost监听。

## 排错与测试

- Turbo未就绪：检查三个权重文件及`turbo_config.json`，重新运行安装脚本。
- CUDA错误或显存不足：检查驱动、其他GPU任务，降低分辨率或参考图数量。
- 找不到标准模型Pipeline：安装`requirements.txt`，其中固定了所需Diffusers版本。
- 端口8321被占用：关闭之前的Studio，或在其他端口运行uvicorn。

```powershell
uv pip install --python .venv -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest test_turbo test_moss -v
.\.venv-turbo\Scripts\python.exe check_native_turbo.py
```

macOS/Linux请改用`.venv/bin/python`等路径。GitHub Actions在三个系统运行CPU回归测试，不代表GPU推理通过。参见[实测结果](docs/VALIDATION.md)及[固定版本与详细信息](README.md)。

仓库只包含源码、安装脚本、测试、文档和生成示例，不包含模型权重、虚拟环境、下载的ComfyUI、本地配置、私人图像或诊断输出。模型及依赖遵循各自的上游条款，包括[Qwen Research许可证](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。
