# -*- coding: utf-8 -*-
"""
Local Diffusers Studio — Qwen-Image-2.1 ローカル画像生成 Web アプリ（バックエンド）

起動方法:
    python -m uvicorn app:app --host 127.0.0.1 --port 8321

モデルバックエンド:
    diffusers : Qwen/Qwen-Image-2.1 を QwenImage21Pipeline で直接実行（既定）
    gguf      : abenzerps/Qwen-Image-2.1-Uncensored-GGUF をローカル ComfyUI
                (leejet/ComfyUI-GGUF) 経由で実行

環境変数:
    QWEN_IMAGE_MODEL   diffusers用モデルID (既定: Qwen/Qwen-Image-2.1)
    OFFLOAD=1          VRAM が足りない場合の CPU オフロードを有効化
    MODEL_LOAD_TIMEOUT モデル読み込み待ちのタイムアウト秒 (既定 1800)
    COMFYUI_URL        ComfyUI サーバーURL (既定: http://127.0.0.1:8188)
    COMFY_CLIP_NAME    GGUF用のテキストエンコーダファイル名
                       (既定: qwen3vl_8b_int8_convrot.safetensors)
    COMFY_VAE_NAME     GGUF用のVAEファイル名 (既定: qwen_image_2.1_vae_bf16.safetensors)
    COMFY_GGUF_FILE    量子化指定を無効化するGGUFファイル名の直接指定
"""
import base64
import gc
import inspect
import io
import json
import os
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Literal

# --- HuggingFace モデルキャッシュをプロジェクト配下（models/hf-cache）へ向ける ---
# diffusers / transformers は import 時に環境変数を参照するため、サードパーティ import より前に設定する
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_BASE_DIR, "models", "hf-cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_BASE_DIR, "models", "hf-cache", "hub"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image-2.1")
MAX_IMAGES = 10
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# --- GGUF / ComfyUI バックエンド設定 ---
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
GGUF_QUANTS = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
DEFAULT_CLIP_NAME = os.environ.get("COMFY_CLIP_NAME", "qwen3vl_8b_int8_convrot.safetensors")
DEFAULT_VAE_NAME = os.environ.get("COMFY_VAE_NAME", "qwen_image_2.1_vae_bf16.safetensors")

# ---------------------------------------------------------------------------
# グローバル状態
# ---------------------------------------------------------------------------
STATE: Dict = {
    "pipe": None,          # QwenImage21Pipeline (diffusersバックエンド用)
    "status": "idle",      # idle | loading | ready | error
    "error": None,         # 読み込みエラーメッセージ
    "device": None,        # cuda / cpu
    "dtype": None,         # torch.bfloat16 など
    "backend": None,
}

JOBS: Dict[str, "queue.Queue"] = {}   # job_id -> SSE イベントキュー
GEN_LOCK = threading.Lock()           # GPU は同時生成 1 件まで


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    images: List[str] = []            # base64 (dataURL も可) 最大10枚
    width: int = Field(2048, ge=256, le=4096)
    height: int = Field(2048, ge=256, le=4096)
    steps: int = Field(40, ge=1, le=300)
    cfg: float = Field(1.0, ge=0.0, le=30.0)
    seed: Optional[int] = -1          # -1 以下はランダム
    backend: Literal["diffusers", "gguf", "turbo"] = "diffusers"
    quant: str = "Q4_K_M"             # GGUF量子化 (backend=gguf のみ)


# ---------------------------------------------------------------------------
# モデル読み込み（バックグラウンドスレッド）— diffusers バックエンド用
# ---------------------------------------------------------------------------
def load_model() -> None:
    STATE.update(status="loading", error=None, backend="diffusers")
    try:
        import torch
        from diffusers import QwenImage21Pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"[studio] モデルを読み込み中: {MODEL_ID} (device={device})")
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if device == "cuda":
            pipe.to("cuda")
            if os.environ.get("OFFLOAD") == "1":
                pipe.enable_model_cpu_offload()
        else:
            print("[studio][warn] CUDA 未検出。CPU で実行します（非常に遅い）")

        STATE.update(pipe=pipe, status="ready", device=device, dtype=str(dtype))
        print(f"[studio] モデル準備完了 ({device}, {dtype})")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        STATE.update(status="error", error=msg)
        print(f"[studio][error] モデル読み込み失敗: {msg}")


@asynccontextmanager
async def lifespan(_app):
    # Load only the selected backend, under GEN_LOCK, when the first job arrives.
    yield


app = FastAPI(title="Local Diffusers Studio", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------
def _decode_image(b64: str):
    from PIL import Image

    raw = b64.split(",", 1)[1] if b64.startswith("data:") else b64
    img = Image.open(io.BytesIO(base64.b64decode(raw)))
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def _http_json(url: str, payload: Optional[dict] = None, timeout: float = 10.0):
    """GET / POST(JSON) を urllib で行い、JSONを返す。"""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_bytes(url: str, timeout: float = 60.0):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _multipart_upload(url: str, field: str, filename: str, data: bytes, extra: Dict[str, str]):
    """簡易 multipart/form-data 送信（urllib）。"""
    boundary = uuid.uuid4().hex
    lines = []
    for k, v in extra.items():
        lines.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    head = (f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode()
    lines.append(head)
    lines.append(data)
    lines.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(lines)
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# diffusers バックエンド（QwenImage21Pipeline）
# ---------------------------------------------------------------------------
def _supported_optionals(pipe) -> Dict[str, bool]:
    """パイプラインが受け付ける任意引数を検査してキャッシュする。"""
    global _SUPPORTED_CACHE  # type: ignore[name-defined]
    try:
        if _SUPPORTED_CACHE is not None:  # type: ignore[name-defined]
            return _SUPPORTED_CACHE  # type: ignore[name-defined]
    except NameError:
        pass

    keys = ("negative_prompt", "guidance_scale", "callback_on_step_end")
    try:
        sig = inspect.signature(pipe.__call__)
        names = set(sig.parameters)
        var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        sup = {k: (var_kw or k in names) for k in keys}
    except Exception:  # noqa: BLE001
        sup = {k: True for k in keys}

    _SUPPORTED_CACHE = sup  # type: ignore[name-defined]
    return sup


def wait_for_ready(q: "queue.Queue") -> None:
    timeout = float(os.environ.get("MODEL_LOAD_TIMEOUT", "1800"))
    t0 = time.time()
    last_msg = None
    while True:
        st = STATE["status"]
        if st == "ready":
            return
        if st == "error":
            raise RuntimeError(STATE["error"] or "モデルの読み込みに失敗しました")
        msg = "モデルを読み込んでいます（初回は重みのダウンロードを含むため時間がかかります）"
        if msg != last_msg:
            q.put({"type": "status", "message": msg})
            last_msg = msg
        if time.time() - t0 > timeout:
            raise TimeoutError("モデルの読み込みに時間がかかりすぎました")
        time.sleep(2)


def run_diffusers_job(job_id: str, params: GenerateRequest) -> None:
    q = JOBS[job_id]
    try:
        if STATE["pipe"] is None:
            load_model()
        wait_for_ready(q)
        pipe = STATE["pipe"]
        device = STATE["device"] or "cuda"

        import torch

        # --- 参照画像のデコード ---
        images = None
        if params.images:
            pil_images = [_decode_image(b) for b in params.images[:MAX_IMAGES]]
            images = pil_images[0] if len(pil_images) == 1 else pil_images

        # --- シード ---
        generator = None
        if params.seed is not None and params.seed >= 0:
            generator = torch.Generator(device).manual_seed(params.seed % (2 ** 31))

        def on_step(_p, step_index, _timestep, callback_kwargs):
            q.put({"type": "step", "step": step_index + 1, "total": params.steps})
            return callback_kwargs

        kwargs: Dict = dict(
            prompt=params.prompt,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
        )
        if images is not None:
            kwargs["image"] = images
        if generator is not None:
            kwargs["generator"] = generator

        sup = _supported_optionals(pipe)
        optional: Dict = {}
        if params.negative_prompt and sup.get("negative_prompt"):
            optional["negative_prompt"] = params.negative_prompt
        if sup.get("guidance_scale"):
            optional["guidance_scale"] = params.cfg
        if sup.get("callback_on_step_end"):
            optional["callback_on_step_end"] = on_step

        # 想定外の非対応引数がある場合は段階的に落として再試行
        attempts = [dict(optional)]
        for drop in (("negative_prompt",), ("guidance_scale",), ("callback_on_step_end",)):
            d = dict(optional)
            d.pop(drop[0], None)
            if d not in attempts:
                attempts.append(d)

        t_start = time.time()
        last_err = None
        for opt in attempts:
            try:
                result = pipe(**kwargs, **opt)
                break
            except TypeError as e:
                last_err = e
                continue
        else:
            raise RuntimeError(f"パイプライン呼び出しに失敗しました: {last_err}")

        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        q.put({
            "type": "done",
            "image_b64": b64,
            "meta": {
                "width": params.width,
                "height": params.height,
                "steps": params.steps,
                "seed": None if (params.seed is None or params.seed < 0) else params.seed % (2 ** 31),
                "elapsed_sec": round(time.time() - t_start, 1),
                "mode": "edit" if images is not None else "t2i",
                "backend": "diffusers",
            },
        })
    except Exception as e:  # noqa: BLE001
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# GGUF バックエンド（ComfyUI + leejet/ComfyUI-GGUF）
# ---------------------------------------------------------------------------
def comfyui_reachable():
    """(接続可否, ComfyUIバージョン) を返す。"""
    try:
        data = _http_json(COMFYUI_URL + "/system_stats", timeout=2.0)
        return True, data.get("system", {}).get("comfyui_version", "?")
    except Exception:  # noqa: BLE001
        return False, None


def build_gguf_workflow(params: GenerateRequest, uploaded_names: List[str], seed: int) -> Dict:
    """ComfyUI API形式のワークフローを構築する。

    ノード構成は AlperKTS/Qwen-Image-2.1-GGUF の動作確認済みワークフローに準拠:
      UnetLoaderGGUF (+QwenImage21Cache[編集時]) → KSampler(euler/simple) → VAEDecode
      CLIPLoader(type=qwen_image) + TextEncodeQwenImage21 → positive/negative/latent
    """
    quant = params.quant if params.quant in GGUF_QUANTS else "Q4_K_M"
    gguf_file = os.environ.get("COMFY_GGUF_FILE") or f"qwen-image-2.1-{quant}.gguf"

    wf: Dict[str, dict] = {
        "unet": {"class_type": "UnetLoaderGGUF",
                 "inputs": {"unet_name": gguf_file}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": DEFAULT_CLIP_NAME, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader",
                "inputs": {"vae_name": DEFAULT_VAE_NAME}},
    }

    cond_inputs: Dict = {
        "clip": ["clip", 0],
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt or "",
    }

    if uploaded_names:
        # --- 編集モード: 参照画像を TextEncodeQwenImage21 に渡し、その latent をキャンバスに ---
        wf["cache"] = {"class_type": "QwenImage21Cache",
                       "inputs": {"model": ["unet", 0], "device": "auto", "dtype": "default"}}
        cond_inputs["vae"] = ["vae", 0]
        cond_inputs["resolution"] = 0   # リサイズなし（32の倍数に丸めるのみ）
        for i, name in enumerate(uploaded_names[:MAX_IMAGES]):
            node_id = f"img{i}"
            wf[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond_inputs[f"images.image_{i + 1}"] = [node_id, 0]
        model_src = ["cache", 0]
        latent_src = ["cond", 2]        # エンコードlatent（image_1 のサイズに追従）
    else:
        # --- T2Iモード: EmptyLatentImage がキャンバス ---
        cond_inputs["resolution"] = 1024
        wf["latent"] = {"class_type": "EmptyLatentImage",
                        "inputs": {"width": params.width, "height": params.height, "batch_size": 1}}
        model_src = ["unet", 0]
        latent_src = ["latent", 0]

    wf["cond"] = {"class_type": "TextEncodeQwenImage21", "inputs": cond_inputs}
    wf["sampler"] = {"class_type": "KSampler", "inputs": {
        "model": model_src,
        "positive": ["cond", 0],
        "negative": ["cond", 1],
        "latent_image": latent_src,
        "seed": seed,
        "steps": params.steps,
        "cfg": params.cfg,
        "sampler_name": "euler",
        "scheduler": "simple",
    }}
    wf["decode"] = {"class_type": "VAEDecode",
                    "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    wf["save"] = {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "qwen_studio", "images": ["decode", 0]}}
    return wf


def run_gguf_job(job_id: str, params: GenerateRequest) -> None:
    q = JOBS[job_id]
    t_start = time.time()
    try:
        ok, ver = comfyui_reachable()
        if not ok:
            raise RuntimeError(
                f"ComfyUI ({COMFYUI_URL}) に接続できません。"
                "ComfyUI を起動し、leejet/ComfyUI-GGUF とモデルファイルを配置してください（README 参照）"
            )

        try:
            import websocket  # websocket-client
        except ImportError:
            raise RuntimeError("websocket-client が未インストールです: pip install websocket-client")

        # --- 参照画像を ComfyUI にアップロード ---
        uploaded = []
        for b64 in params.images[:MAX_IMAGES]:
            raw = b64.split(",", 1)[1] if b64.startswith("data:") else b64
            data = base64.b64decode(raw)
            res = _multipart_upload(
                COMFYUI_URL + "/upload/image", "image",
                f"ref_{uuid.uuid4().hex[:8]}.png", data, {"overwrite": "true"},
            )
            uploaded.append(res["name"])

        seed = (params.seed % (2 ** 31)) if (params.seed is not None and params.seed >= 0) \
            else random.randint(0, 2 ** 31 - 1)

        wf = build_gguf_workflow(params, uploaded, seed)
        prompt_id = _http_json(COMFYUI_URL + "/prompt", {"prompt": wf})["prompt_id"]
        q.put({"type": "status", "message": f"ComfyUI で生成中 (v{ver})…"})

        # --- WebSocket で進捗を受信 ---
        ws_url = COMFYUI_URL.replace("http://", "ws://").replace("https://", "wss://") \
            + f"/ws?clientId={uuid.uuid4().hex}"
        ws = websocket.create_connection(ws_url, timeout=1800)
        try:
            while True:
                msg = json.loads(ws.recv())
                mtype = msg.get("type")
                data = msg.get("data", {})
                if mtype == "progress":
                    q.put({"type": "step", "step": int(data.get("value", 0)),
                           "total": int(data.get("max", params.steps))})
                elif mtype == "executing" and data.get("node") is None:
                    break
                elif mtype == "execution_error":
                    raise RuntimeError(f"ComfyUI エラー: {data.get('exception_message', '不明')}")
        finally:
            ws.close()

        # --- 結果画像を取得 ---
        hist = _http_json(COMFYUI_URL + f"/history/{prompt_id}", timeout=30.0)
        entry = hist.get(prompt_id, {})
        if not entry.get("status", {}).get("completed", False):
            raise RuntimeError(f"ComfyUI ジョブが完了しませんでした: {entry.get('status')}")

        out_images = []
        for node_out in entry.get("outputs", {}).values():
            for img in node_out.get("images", []):
                out_images.append(img)
        if not out_images:
            raise RuntimeError("ComfyUI の出力画像が見つかりませんでした")

        first = out_images[0]
        view_url = (COMFYUI_URL + "/view?filename=" + urllib.request.quote(first["filename"])
                    + "&subfolder=" + urllib.request.quote(first.get("subfolder", ""))
                    + "&type=" + first.get("type", "output"))
        img_bytes = _http_bytes(view_url)

        q.put({
            "type": "done",
            "image_b64": base64.b64encode(img_bytes).decode("ascii"),
            "meta": {
                "width": params.width,
                "height": params.height,
                "steps": params.steps,
                "seed": seed,
                "elapsed_sec": round(time.time() - t_start, 1),
                "mode": "edit" if uploaded else "t2i",
                "backend": f"gguf ({params.quant})",
            },
        })
    except Exception as e:  # noqa: BLE001
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# ジョブディスパッチ
# ---------------------------------------------------------------------------
def unload_diffusers() -> None:
    """Release only this app's pipeline before using another GPU backend."""
    pipe = STATE.pop("pipe", None)
    STATE["pipe"] = None
    if pipe is not None:
        del pipe
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    STATE.update(status="idle", error=None, device=None, dtype=None)


def turbo_config() -> Dict:
    path = Path(_BASE_DIR) / "turbo_config.json"
    if not path.is_file():
        raise RuntimeError("Turbo の準備が必要です。setup_turbo.ps1 を実行してください")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("python", "comfy", "transformer", "encoder", "vae"):
        if not Path(config[key]).exists():
            raise RuntimeError(f"Turbo {key} が見つかりません: {config[key]} (setup_turbo.ps1 を再実行)")
    return config


def run_turbo_job(job_id: str, params: GenerateRequest) -> None:
    q = JOBS[job_id]
    started = time.time()
    try:
        STATE.update(backend="turbo", status="loading", error=None, device="cuda", dtype="INT8 ConvRot")
        config = turbo_config()
        seed = params.seed if params.seed is not None and params.seed >= 0 else random.randint(0, 2 ** 31 - 1)
        with tempfile.TemporaryDirectory(prefix="qwen-turbo-") as directory:
            request = Path(directory) / "request.json"
            output = Path(directory) / "result.png"
            request.write_text(json.dumps({**params.model_dump(), "seed": seed, "config": config}), encoding="utf-8")
            q.put({"type": "status", "message": "Turbo INT8 ConvRot を読み込み中…"})
            log = []
            with subprocess.Popen(
                [config["python"], "-u", str(Path(_BASE_DIR) / "turbo_worker.py"), str(request), str(output)],
                cwd=_BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            ) as worker:
                for line in worker.stdout:
                    if line.startswith("STUDIO_EVENT "):
                        event = json.loads(line[len("STUDIO_EVENT "):])
                        if event.get("type") == "ready":
                            STATE["status"] = "ready"
                        else:
                            q.put(event)
                    else:
                        log.append(line.rstrip())
                        log = log[-12:]
                        encoding = sys.stdout.encoding or "utf-8"
                        safe_line = line.rstrip().encode(encoding, errors="backslashreplace").decode(encoding)
                        print("[turbo]", safe_line, flush=True)
                code = worker.wait()
            if code != 0 or not output.is_file():
                raise RuntimeError(f"Turbo worker failed (exit {code}): " +
                                   ("\n".join(log)[-3000:] or "出力画像がありません。workerが中断されました"))
            from PIL import Image
            with Image.open(output) as image:
                width, height = image.size
            q.put({"type": "done", "image_b64": base64.b64encode(output.read_bytes()).decode("ascii"), "meta": {
                "width": width, "height": height, "steps": 8, "cfg": 1.0, "seed": seed,
                "elapsed_sec": round(time.time() - started, 1),
                "mode": "edit" if params.images else "t2i", "backend": "turbo INT8 ConvRot",
            }})
        STATE.update(status="idle", device=None, dtype=None)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        STATE.update(status="error", error=message)
        q.put({"type": "error", "message": message})


def run_generation(job_id: str, params: GenerateRequest) -> None:
    try:
        if params.backend != "diffusers":
            unload_diffusers()
        if params.backend == "turbo":
            run_turbo_job(job_id, params)
        elif params.backend == "gguf":
            STATE.update(backend="gguf", status="idle", error=None)
            run_gguf_job(job_id, params)
        else:
            run_diffusers_job(job_id, params)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        STATE.update(status="error", error=message)
        JOBS[job_id].put({"type": "error", "message": message})
    finally:
        GEN_LOCK.release()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status():
    out = {
        "model_id": MODEL_ID,
        "status": STATE["status"],
        "error": STATE["error"],
        "device": STATE["device"],
        "dtype": (STATE["dtype"] or "").replace("torch.", ""),
        "backend": STATE["backend"],
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out.update({
                "gpu_name": props.name,
                "vram_total_gb": round(props.total_memory / 1024 ** 3, 1),
                "vram_used_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 1),
            })
    except Exception:  # noqa: BLE001
        pass

    comfy_ok, comfy_ver = comfyui_reachable()
    out["comfyui"] = {
        "url": COMFYUI_URL,
        "reachable": comfy_ok,
        "version": comfy_ver,
    }
    try:
        turbo_config()
        out["turbo"] = {"available": True, "steps": 8, "cfg": 1.0}
    except Exception as exc:
        out["turbo"] = {"available": False, "error": str(exc)}
    return out


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="プロンプトを入力してください")

    if req.backend == "gguf" and req.quant not in GGUF_QUANTS:
        raise HTTPException(status_code=422, detail=f"不明な量子化です: {req.quant}")
    if req.backend == "turbo":
        if not req.images and (req.width % 32 or req.height % 32):
            raise HTTPException(status_code=422, detail="Turbo の幅・高さは32の倍数を指定してください")
        req = req.model_copy(update={"steps": 8, "cfg": 1.0})

    if not GEN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="別の生成が進行中です。完了をお待ちください")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = queue.Queue()
    threading.Thread(target=run_generation, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/events")
def api_job_events(job_id: str):
    q = JOBS.get(job_id)
    if q is None:
        raise HTTPException(status_code=404, detail="不明なジョブです")

    def stream():
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item["type"] in ("done", "error"):
                    break
        finally:
            JOBS.pop(job_id, None)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 静的ファイル（API ルートより後でマウント）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
