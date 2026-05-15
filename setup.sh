#!/usr/bin/env bash
# =============================================================
#  setup.sh  —  One-click environment installer (Linux / Mac)
#  Usage: bash setup.sh
# =============================================================
set -e

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "  GCN-BiLSTM-BAN  —  Environment Setup"
echo "============================================================"
echo ""

# ── 1. Check conda ────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    error "conda not found. Please install Miniconda first:
       https://docs.conda.io/en/latest/miniconda.html
  Then open a NEW terminal and run this script again."
fi
CONDA_BASE=$(conda info --base)
info "conda found at: $CONDA_BASE"

# ── 2. Load conda shell functions ────────────────────────────
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 3. Check if env already exists ───────────────────────────
if conda env list | grep -q "^drugban "; then
    warn "Conda environment 'drugban' already exists."
    read -rp "  Re-create it from scratch? (y/n) [n]: " REPLY
    REPLY=${REPLY:-n}
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        info "Removing old environment..."
        conda env remove -n drugban -y
    else
        info "Keeping existing environment. Activating to verify..."
        conda activate drugban
        python - <<'PY'
import sys
missing = []
for pkg in ['torch','dgl','dgllife','rdkit','pandas','sklearn','yacs']:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
if missing:
    print(f"Missing packages: {missing}")
    sys.exit(1)
else:
    print("All packages present.")
PY
        echo ""
        success "Environment 'drugban' is ready.  Run:  conda activate drugban"
        exit 0
    fi
fi

# ── 4. Create environment ─────────────────────────────────────
info "Creating conda environment 'drugban' (this takes 5–15 minutes)..."
conda env create -f environment.yml
success "Environment created."

# ── 5. Activate and verify ────────────────────────────────────
info "Activating environment and verifying installation..."
conda activate drugban

python - <<'PY'
import sys
print("  Checking packages...")
ok = True
checks = {
    'torch':     'PyTorch',
    'dgl':       'DGL',
    'dgllife':   'DGL-Life',
    'rdkit':     'RDKit',
    'pandas':    'Pandas',
    'sklearn':   'scikit-learn',
    'yacs':      'YACS',
    'tqdm':      'tqdm',
}
for mod, name in checks.items():
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'unknown')
        print(f"    {name}: {ver}")
    except ImportError:
        print(f"    {name}: MISSING  <--- ERROR")
        ok = False

import torch
cuda = torch.cuda.is_available()
print(f"  GPU (CUDA) available: {cuda}")
if not cuda:
    print("  (CPU-only mode — predictions will be ~10x slower but correct)")

if not ok:
    print("\nERROR: Some packages failed to install. Try running setup.sh again.")
    sys.exit(1)
PY

echo ""
echo "============================================================"
echo -e "${GREEN}  Setup complete!${NC}"
echo ""
echo "  To start predicting:"
echo "    conda activate drugban"
echo "    python predict_simple.py       # interactive (one pair)"
echo "    python predict_batch.py --help # batch mode (many pairs)"
echo "============================================================"
echo ""
