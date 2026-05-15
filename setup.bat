@echo off
REM =============================================================
REM  setup.bat  --  One-click environment installer (Windows)
REM  Usage: Double-click this file, OR run in Anaconda Prompt:
REM         setup.bat
REM =============================================================
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   GCN-BiLSTM-BAN  --  Environment Setup  (Windows)
echo ============================================================
echo.

REM ── 1. Check conda ──────────────────────────────────────────
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda not found.
    echo.
    echo   Please install Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo   After installing, open "Anaconda Prompt" from the Start menu
    echo   and run this script again.
    echo.
    pause
    exit /b 1
)
echo [OK]   conda found.

REM ── 2. Check if environment already exists ──────────────────
conda env list | findstr /B "drugban " >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [WARN] Conda environment 'drugban' already exists.
    set /p REPLY="  Re-create from scratch? (y/n) [n]: "
    if /i "!REPLY!"=="y" (
        echo [INFO] Removing old environment...
        call conda env remove -n drugban -y
    ) else (
        echo [INFO] Keeping existing environment.
        goto :verify
    )
)

REM ── 3. Create environment ────────────────────────────────────
echo [INFO] Creating conda environment 'drugban' ...
echo        (This takes 5-15 minutes on first run)
echo.
call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment creation failed. Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK]   Environment created.

REM ── 4. Verify installation ───────────────────────────────────
:verify
echo [INFO] Activating environment and verifying...
call conda activate drugban
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate environment.
    echo   Try running in Anaconda Prompt: conda activate drugban
    pause
    exit /b 1
)

python -c ^
"import sys; missing=[];" ^
"[missing.append(p) or None for p in ['torch','dgl','dgllife','rdkit','pandas','sklearn','yacs','tqdm'] if not __import__(p, globals(), locals(), [], 0) and missing.append(p)];" ^
"print('  All packages OK') if not missing else (print(f'  Missing: {missing}'), sys.exit(1))"

REM Simpler verification:
python -c "import torch, dgl, dgllife, rdkit, pandas, sklearn, yacs, tqdm; import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package verification failed. Try running setup.bat again.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To start predicting, open Anaconda Prompt and run:
echo     conda activate drugban
echo     python predict_simple.py
echo.
echo   For batch predictions (many pairs at once):
echo     python predict_batch.py --help
echo ============================================================
echo.
pause
