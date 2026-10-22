@echo off
cd /d "%~dp0"
rem Point HuggingFace model cache to the project directory (models\hf-cache)
set "HF_HOME=%~dp0models\hf-cache"
set "HUGGINGFACE_HUB_CACHE=%~dp0models\hf-cache\hub"
rem Prefer dedicated venv (diffusers / CUDA torch are installed in .venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYEXE=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [error] Python not found in PATH, and .venv does not exist.
        pause
        exit /b 1
    )
    set "PYEXE=python"
)
echo Starting Local Diffusers Studio: http://127.0.0.1:8321 (using %PYEXE%)
"%PYEXE%" -m uvicorn app:app --host 127.0.0.1 --port 8321
pause
