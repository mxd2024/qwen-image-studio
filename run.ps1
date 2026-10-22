# Local Diffusers Studio startup script
Set-Location -LiteralPath $PSScriptRoot

# Point HuggingFace model cache to the project directory (models\hf-cache)
$env:HF_HOME = Join-Path $PSScriptRoot "models\hf-cache"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $PSScriptRoot "models\hf-cache\hub"

# Prefer dedicated venv (diffusers / CUDA torch are installed in .venv)
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $py = $venvPy
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host "[error] Python not found in PATH, and .venv does not exist" -ForegroundColor Red
        exit 1
    }
    $py = "python"
}

Write-Host "Starting Local Diffusers Studio: http://127.0.0.1:8321 (using $py)"
& $py -m uvicorn app:app --host 127.0.0.1 --port 8321
