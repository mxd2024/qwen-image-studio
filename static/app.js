/* ============================================================
   Local Diffusers Studio — フロントエンド
   ============================================================ */
"use strict";

const $ = (s) => document.querySelector(s);

// ---------- 定数 ----------
// README推奨サイズ（ネイティブ2K）をローカルGPU向けに半分にしたプリセット。
// 「2K」はフルネイティブ解像度。幅・高さは手動編集自由（16の倍数が望ましい）。
const PRESETS = [
  { id: "1:1",  label: "1:1",  w: 1024, h: 1024 },
  { id: "16:9", label: "16:9", w: 1376, h: 768 },
  { id: "9:16", label: "9:16", w: 768,  h: 1376 },
  { id: "4:3",  label: "4:3",  w: 1200, h: 896 },
  { id: "3:4",  label: "3:4",  w: 896,  h: 1200 },
  { id: "2:3",  label: "2:3",  w: 832,  h: 1248 },
  { id: "2K",   label: "2K",   w: 2048, h: 2048 },
];

const MAX_IMAGES = 10;
const MAX_FILE_MB = 15;

// ---------- 状態 ----------
const state = {
  refs: [],        // [{ name, dataUrl }]
  generating: false,
  es: null,        // EventSource
  t0: 0,           // 生成開始時刻
  timerId: null,
  backend: "diffusers",   // diffusers | turbo | moss | moss_int8
  sampling: { diffusers: { steps: "40", cfg: "1" }, moss: { steps: "50", cfg: "3" },
    moss_int8: { steps: "50", cfg: "3" } },
};

// ---------- DOM参照 ----------
const elPrompt = $("#prompt");
const elNegWrap = $("#neg-wrap");
const elNegToggle = $("#neg-toggle");
const elDropzone = $("#dropzone");
const elFileInput = $("#file-input");
const elDzEmpty = $("#dz-empty");
const elThumbs = $("#thumbs");
const elRefCount = $("#ref-count");
const elPresets = $("#presets");
const elModelOpts = $("#model-opts");
const elWidth = $("#width");
const elHeight = $("#height");
const elSteps = $("#steps");
const elCfg = $("#cfg");
const elSeed = $("#seed");
const elBtn = $("#btn-generate");
const elBtnLabel = $("#btn-label");
const elBtnArrow = $("#btn-arrow");
const elBtnSpinner = $("#btn-spinner");
const elModeBadge = $("#mode-badge");
const elDownload = $("#download");
const elPvPlaceholder = $("#pv-placeholder");
const elResultImg = $("#result-img");
const elPvLoading = $("#pv-loading");
const elLoadingText = $("#loading-text");
const elStatusBar = $("#status-bar");
const elElapsedBox = $("#elapsed-box");
const elToast = $("#toast");

// ============================================================
// 初期化
// ============================================================
function init() {
  buildPresets();
  bindModelOpts();
  bindDiceIcons();
  bindEvents();
  pollStatus();
  setInterval(pollStatus, 3000);
}

// モデルバックエンド切替
function bindModelOpts() {
  elModelOpts.querySelectorAll(".mopt").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (state.generating || state.backend === btn.dataset.backend) return;
      if (state.backend !== "turbo") {
        state.sampling[state.backend] = { steps: elSteps.value, cfg: elCfg.value };
      }
      state.backend = btn.dataset.backend;
      const turbo = state.backend === "turbo";
      const moss = state.backend.startsWith("moss");
      const sampling = turbo ? { steps: "8", cfg: "1" } : state.sampling[state.backend];
      elSteps.value = sampling.steps;
      elCfg.value = sampling.cfg;
      [elSteps, elCfg, ...document.querySelectorAll('.dice[data-target="steps"], .dice[data-target="cfg"]')]
        .forEach((el) => { el.disabled = turbo; });
      [elWidth, elHeight].forEach((el) => { el.step = turbo ? 32 : moss ? 16 : 8; });
      $("#turbo-note").classList.toggle("hidden", !turbo);
      $("#turbo-status").classList.toggle("hidden", !turbo);
      $("#moss-note").classList.toggle("hidden", !moss);
      $("#moss-status").classList.toggle("hidden", !moss);
      elDropzone.setAttribute("aria-disabled", String(moss));
      elFileInput.disabled = moss;
      renderRefs();
      elModelOpts.querySelectorAll(".mopt").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      pollStatus();
    });
  });
}

// プリセットピル生成
function buildPresets() {
  elPresets.innerHTML = "";
  PRESETS.forEach((p) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "pill" + (p.id === "1:1" ? " active" : "");
    b.textContent = p.label;
    b.dataset.id = p.id;
    b.addEventListener("click", () => {
      elWidth.value = state.backend === "turbo" ? Math.round(p.w / 32) * 32 : p.w;
      elHeight.value = state.backend === "turbo" ? Math.round(p.h / 32) * 32 : p.h;
      setActivePill(p.id);
    });
    elPresets.appendChild(b);
  });
}

function setActivePill(id) {
  elPresets.querySelectorAll(".pill").forEach((p) =>
    p.classList.toggle("active", p.dataset.id === id)
  );
}

// ダイスアイコン（SVG）を各ボタンに挿入
const DICE_SVG = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
  <rect x="3.5" y="3.5" width="17" height="17" rx="4"/>
  <circle cx="9" cy="9" r="1.4" fill="currentColor" stroke="none"/>
  <circle cx="15" cy="15" r="1.4" fill="currentColor" stroke="none"/>
</svg>`;

function bindDiceIcons() {
  document.querySelectorAll(".dice").forEach((btn) => {
    btn.innerHTML = DICE_SVG;
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      randomize(btn.dataset.target);
    });
  });
}

// ランダム化（ダイス）
function randomize(target) {
  const ri = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
  switch (target) {
    case "size": {
      const p = PRESETS[ri(0, PRESETS.length - 1)];
      elWidth.value = state.backend === "turbo" ? Math.round(p.w / 32) * 32 : p.w;
      elHeight.value = state.backend === "turbo" ? Math.round(p.h / 32) * 32 : p.h;
      setActivePill(p.id);
      break;
    }
    case "steps":
      elSteps.value = ri(20, 60);
      break;
    case "cfg": {
      const opts = [1, 1.5, 2, 3, 4, 5, 7];
      elCfg.value = opts[ri(0, opts.length - 1)];
      break;
    }
    case "seed":
      elSeed.value = ri(0, 999999999);
      break;
  }
}

// ============================================================
// イベントバインド
// ============================================================
function bindEvents() {
  // ネガティブプロンプトの折りたたみ
  elNegToggle.addEventListener("click", () => {
    const open = elNegWrap.classList.toggle("hidden");
    elNegToggle.classList.toggle("open", !open);
  });

  // ドロップゾーン: クリックでファイル選択
  elDropzone.addEventListener("click", (e) => {
    if (state.backend.startsWith("moss")) return;
    if (e.target.closest(".thumb")) return; // サムネイル操作は除外
    elFileInput.click();
  });

  elFileInput.addEventListener("change", () => {
    addFiles(elFileInput.files);
    elFileInput.value = "";
  });

  // ドラッグ&ドロップ
  ["dragenter", "dragover"].forEach((ev) =>
    elDropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      elDropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    elDropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      elDropzone.classList.remove("dragover");
    })
  );
  elDropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  // 幅/高さの手動編集時はピセットのアクティブ解除
  [elWidth, elHeight].forEach((inp) =>
    inp.addEventListener("input", () => setActivePill(""))
  );

  // 生成ボタン
  elBtn.addEventListener("click", generate);
}

// ============================================================
// 参照画像の管理
// ============================================================
async function addFiles(fileList) {
  if (state.backend.startsWith("moss")) {
    toast("MOSSはテキスト／レイアウト生成専用です。画像編集にはQwenを選択してください", true);
    return;
  }
  const files = Array.from(fileList || []);
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    if (state.refs.length >= MAX_IMAGES) {
      toast(`最大${MAX_IMAGES}枚までです`, true);
      break;
    }
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      toast(`${f.name}: ${MAX_FILE_MB}MBを超えるファイルはスキップしました`, true);
      continue;
    }
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error(`${f.name}: 画像を読み込めませんでした`));
        reader.readAsDataURL(f);
      });
      state.refs.push({ name: f.name, dataUrl });
      renderRefs();
    } catch (error) {
      toast(error.message, true);
    }
  }
}

function removeRef(i) {
  state.refs.splice(i, 1);
  renderRefs();
}

function renderRefs() {
  const n = state.refs.length;
  elRefCount.textContent = state.backend.startsWith("moss") ? "MOSSはテキスト／レイアウト生成専用" :
    n > 0 ? `${n}/${MAX_IMAGES}枚` : "任意・最大10枚";
  elDzEmpty.classList.toggle("hidden", n > 0);
  elThumbs.classList.toggle("hidden", n === 0);

  elThumbs.innerHTML = "";
  state.refs.forEach((r, i) => {
    const d = document.createElement("div");
    d.className = "thumb";
    d.innerHTML = `
      <img src="${r.dataUrl}" alt="" />
      <span class="idx">${i + 1}</span>
      <button class="x" title="削除">&times;</button>`;
    d.querySelector(".x").addEventListener("click", (e) => {
      e.stopPropagation();
      removeRef(i);
    });
    elThumbs.appendChild(d);
  });

  // モードバッジ切替
  const edit = n > 0 && !state.backend.startsWith("moss");
  elModeBadge.textContent = edit ? "IMAGE EDIT" : "TEXT TO IMAGE";
  elModeBadge.classList.toggle("edit", edit);
}

// ============================================================
// ステータス取得（フッター表示）
// ============================================================
async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const s = await res.json();
    const dev = s.device ? `${s.device === "cuda" ? "cuda:0" : "cpu"} · ${(s.dtype || "?").toUpperCase()}` : "—";
    let st;
    switch (s.status) {
      case "ready":   st = "準備完了"; break;
      case "loading": st = "モデル読み込み中…"; break;
      case "error":   st = `エラー: ${s.error || ""}`.slice(0, 80); break;
      default:        st = "待機中";
    }
    $("#turbo-status").textContent = s.backend === "turbo" && s.status === "error"
      ? `Turbo エラー: ${s.error}`
      : s.turbo?.available
      ? "Turbo 利用可能 · ローカル実行"
      : `Turbo 準備が必要: ${s.turbo?.error || "設定を確認してください"}`;
    const mossBackend = state.backend === "moss_int8" ? "moss_int8" : "moss";
    $("#moss-status").textContent = s.backend === mossBackend && s.status === "error"
      ? `MOSS エラー: ${s.error}`
      : s[mossBackend]?.available ? "MOSS 利用可能 · ローカル実行"
      : `MOSS 準備が必要: ${s[mossBackend]?.error || "設定を確認してください"}`;
    elStatusBar.textContent = `${dev} · ${st}${s.gpu_name ? ` · ${s.gpu_name}` : ""}`;
  } catch {
    /* サーバー未起動時は無視 */
  }
}

// ============================================================
// 生成フロー
// ============================================================
async function generate() {
  if (state.generating) return;

  const prompt = elPrompt.value.trim();
  if (!prompt) {
    toast("プロンプトを入力してください", true);
    elPrompt.focus();
    return;
  }

  const payload = {
    prompt,
    negative_prompt: $("#negative-prompt").value.trim() || null,
    images: state.backend.startsWith("moss") ? [] : state.refs.map((r) => r.dataUrl),
    width: clampInt(elWidth.value, 256, 4096, 1024),
    height: clampInt(elHeight.value, 256, 4096, 1024),
    steps: clampInt(elSteps.value, 1, 300, 40),
    cfg: Math.min(30, Math.max(0, Number.isNaN(parseFloat(elCfg.value)) ? 1 : parseFloat(elCfg.value))),
    seed: Number.isNaN(parseInt(elSeed.value, 10)) ? -1 : parseInt(elSeed.value, 10),
    backend: state.backend,
  };

  if (state.backend === "turbo" && !state.refs.length && (payload.width % 32 || payload.height % 32)) {
    toast("Turbo の幅・高さは32の倍数を指定してください", true);
    return;
  }
  if (state.backend.startsWith("moss") && (payload.width % 16 || payload.height % 16)) {
    toast("MOSS の幅・高さは16の倍数を指定してください", true);
    return;
  }

  setGenerating(true);
  elPvLoading.classList.remove("hidden");
  elResultImg.classList.add("hidden");
  elDownload.classList.add("hidden");
  setLoadingText("リクエスト送信中…");
  state.t0 = Date.now();
  startElapsedTimer();

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.status === 409) {
      toast("別の生成が進行中です。完了をお待ちください", true);
      setGenerating(false);
      stopElapsedTimer();
      elPvLoading.classList.add("hidden");
      return;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const { job_id } = await res.json();
    connectSSE(job_id);
  } catch (e) {
    fail(e.message || String(e));
  }
}

function connectSSE(jobId) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.es = es;

  es.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }

    switch (msg.type) {
      case "status":
        setLoadingText(msg.message || "待機中…");
        break;
      case "step":
        setLoadingText(`ステップ ${msg.step} / ${msg.total}`);
        break;
      case "done":
        onDone(msg);
        es.close();
        state.es = null;
        break;
      case "error":
        fail(msg.message || "生成に失敗しました");
        es.close();
        state.es = null;
        break;
    }
  };

  es.onerror = () => {
    // サーバー再起動等で切断された場合（done/error受信後は無視）
    if (state.generating) {
      fail("サーバーとの接続が切れました");
      es.close();
      state.es = null;
    }
  };
}

function onDone(msg) {
  const dataUrl = `data:image/png;base64,${msg.image_b64}`;
  elResultImg.src = dataUrl;
  elResultImg.classList.remove("hidden");
  elPvPlaceholder.classList.add("hidden");
  elPvLoading.classList.add("hidden");

  elDownload.href = dataUrl;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const prefix = msg.meta?.backend?.startsWith("MOSS") ? "moss-image" : "qwen-image";
  elDownload.download = `${prefix}-${stamp}.png`;
  elDownload.classList.remove("hidden");

  setGenerating(false);
  stopElapsedTimer();
  if (msg.meta) {
    toast(`生成完了（${msg.meta.elapsed_sec}秒）`);
  }
}

function fail(message) {
  setGenerating(false);
  stopElapsedTimer();
  elPvLoading.classList.add("hidden");
  toast(message, true);
}

// ============================================================
// UIヘルパー
// ============================================================
function setGenerating(on) {
  state.generating = on;
  elBtn.disabled = on;
  elBtnLabel.textContent = on ? "生成中…" : "生成する";
  elBtnArrow.classList.toggle("hidden", on);
  elBtnSpinner.classList.toggle("hidden", !on);
}

function setLoadingText(t) {
  elLoadingText.textContent = t;
}

function startElapsedTimer() {
  stopElapsedTimer();
  state.timerId = setInterval(() => {
    const sec = Math.floor((Date.now() - state.t0) / 1000);
    elElapsedBox.textContent = `${sec}s`;
  }, 500);
}

function stopElapsedTimer() {
  if (state.timerId) clearInterval(state.timerId);
  state.timerId = null;
  elElapsedBox.textContent = "";
}

let toastTimer = null;
function toast(message, isErr = false) {
  elToast.textContent = message;
  elToast.classList.toggle("err", isErr);
  elToast.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elToast.classList.remove("show"), 3200);
}

function clampInt(v, min, max, fallback) {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

init();
