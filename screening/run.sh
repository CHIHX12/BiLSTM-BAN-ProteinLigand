#!/usr/bin/env bash
# =============================================================================
#  HTS Drug-Protein Binding Screening — One-Click Run Script
#  run.sh
# =============================================================================
#
#  Usage:
#    chmod +x run.sh          # first time only
#    ./run.sh                 # standard run (bilstm model)
#    ./run.sh --model cnn     # use CNN model
#    ./run.sh --top 100       # save top-100 results
#    ./run.sh --threshold 0.6 # stricter binding threshold
#    ./run.sh --test          # run test suite first, then screen
#    ./run.sh --test-only     # run test suite only (no screening)
#    ./run.sh --fast-test     # skip slow benchmark tests, then screen
#    ./run.sh --help          # show this help
#
#  Exit codes:
#    0 = Success
#    1 = Validation or screening failed
#    2 = Missing dependency or file
#    3 = Test suite failed
# =============================================================================

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

# ── Locate script directory ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Default parameters ─────────────────────────────────────────────────────
MODEL="bilstm"
TOP=50
THRESHOLD=0.5
BATCH_SIZE=32
LIGANDS="input/ligands/ligands.txt"
PROTEINS="input/proteins/proteins.txt"
RUN_TESTS=false
TEST_ONLY=false
FAST_TEST=false
SKIP_VAL=false
NO_BINDING_SITES=false
EXTRA_ARGS=""

# ── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)         MODEL="$2";        shift 2 ;;
    --top)           TOP="$2";          shift 2 ;;
    --threshold)     THRESHOLD="$2";    shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";   shift 2 ;;
    --ligands)       LIGANDS="$2";      shift 2 ;;
    --proteins)      PROTEINS="$2";     shift 2 ;;
    --test)          RUN_TESTS=true;    shift ;;
    --test-only)     TEST_ONLY=true;    shift ;;
    --fast-test)     FAST_TEST=true; RUN_TESTS=true; shift ;;
    --skip-validation) SKIP_VAL=true;   shift ;;
    --no-binding-sites) NO_BINDING_SITES=true; shift ;;
    --help|-h)
      sed -n '/^#  Usage:/,/^# ====/{ /^# ====/!p }' "$0" | sed 's/^#  \?//'
      exit 0 ;;
    *) echo -e "${RED}Unknown argument: $1${NC}"; exit 1 ;;
  esac
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo -e "${BOLD}=================================================================${NC}"
echo -e "${BOLD}  HTS Drug-Protein Binding Screening${NC}"
echo -e "  ${TIMESTAMP}"
echo -e "${BOLD}=================================================================${NC}"

# ── Step 0: Check Python ───────────────────────────────────────────────────
echo ""
echo -e "${BLUE}[0/5] Environment check${NC}"

if ! command -v python3 &>/dev/null; then
  echo -e "${RED}  ✗ python3 not found. Please activate your conda environment.${NC}"
  echo -e "       conda activate drugban"
  exit 2
fi

PYTHON_VER=$(python3 --version 2>&1)
echo -e "  Python: ${PYTHON_VER}"

# Check GPU
GPU_INFO=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f'GPU: {name} ({mem} GB VRAM)')
else:
    print('CPU only (no CUDA GPU detected)')
" 2>/dev/null || echo "torch not available")

if [[ "$GPU_INFO" == *"CPU only"* ]]; then
  echo -e "${YELLOW}  ⚠  ${GPU_INFO}${NC}"
  echo -e "${YELLOW}     CPU mode is slower. Expect ~1–3 min per 1,000 pairs.${NC}"
else
  echo -e "${GREEN}  ${GPU_INFO}${NC}"
fi

# Check key dependencies
MISSING_DEPS=()
for pkg in torch dgl dgllife rdkit; do
  python3 -c "import $pkg" 2>/dev/null || MISSING_DEPS+=("$pkg")
done

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
  echo -e "${RED}  ✗ Missing dependencies: ${MISSING_DEPS[*]}${NC}"
  echo -e "     Install with:"
  echo -e "       conda install -c dglteam dgl-cuda12.1"
  echo -e "       conda install -c conda-forge rdkit dgllife"
  echo -e "       pip install torch"
  exit 2
fi
echo -e "${GREEN}  ✓ All dependencies present${NC}"

# ── Step 1: Check model checkpoints ───────────────────────────────────────
echo ""
echo -e "${BLUE}[1/5] Model checkpoint check${NC}"

BILSTM_CKPT="../result/DrugBAN_BiLSTM/best_model_epoch_94.pth"
CNN_CKPT="../result/DrugBAN/best_model_epoch_90.pth"

check_ckpt() {
  local path="$1" label="$2"
  if [[ -f "$path" ]]; then
    local size=$(du -h "$path" | cut -f1)
    echo -e "${GREEN}  ✓ ${label}: ${path} (${size})${NC}"
    return 0
  else
    echo -e "${YELLOW}  ⚠ ${label}: NOT FOUND at ${path}${NC}"
    return 1
  fi
}

bilstm_ok=true; cnn_ok=true
check_ckpt "$BILSTM_CKPT" "BiLSTM checkpoint" || bilstm_ok=false
check_ckpt "$CNN_CKPT"    "CNN checkpoint"    || cnn_ok=false

if [[ "$MODEL" == "bilstm" && "$bilstm_ok" == "false" ]]; then
  echo -e "${RED}  ✗ BiLSTM checkpoint required but missing. Cannot continue.${NC}"
  exit 2
fi
if [[ "$MODEL" == "cnn" && "$cnn_ok" == "false" ]]; then
  echo -e "${RED}  ✗ CNN checkpoint required but missing. Cannot continue.${NC}"
  exit 2
fi

# ── Step 2: Check input files ──────────────────────────────────────────────
echo ""
echo -e "${BLUE}[2/5] Input file check${NC}"

check_input() {
  local path="$1" label="$2"
  if [[ ! -f "$path" ]]; then
    echo -e "${RED}  ✗ ${label} not found: ${path}${NC}"
    return 1
  fi
  # Count non-comment, non-blank lines
  local count
  count=$(grep -v '^#' "$path" | grep -v '^[[:space:]]*$' | wc -l || true)
  if [[ "$count" -eq 0 ]]; then
    echo -e "${RED}  ✗ ${label} is empty (only comments/blanks): ${path}${NC}"
    echo -e "     Please add your drug/protein data to this file."
    return 1
  fi
  echo -e "${GREEN}  ✓ ${label}: ${count} entries  (${path})${NC}"
  return 0
}

inputs_ok=true
check_input "$LIGANDS"  "Ligands file"  || inputs_ok=false
check_input "$PROTEINS" "Proteins file" || inputs_ok=false

if [[ "$inputs_ok" == "false" ]]; then
  echo ""
  echo -e "${YELLOW}  Please fill in your input files and retry.${NC}"
  echo -e "  See input/ligands/ligands_template.txt for format examples."
  exit 2
fi

# ── Step 3 (Optional): Run tests ───────────────────────────────────────────
if [[ "$RUN_TESTS" == "true" || "$TEST_ONLY" == "true" ]]; then
  echo ""
  echo -e "${BLUE}[3/5] Running test suite${NC}"

  TEST_ARGS=""
  if [[ "$FAST_TEST" == "true" ]]; then
    TEST_ARGS="--fast"
    echo -e "  Mode: fast (skipping model benchmark)"
  else
    echo -e "  Mode: full (including model benchmark on BindingDB)"
  fi

  if python3 run_tests.py $TEST_ARGS; then
    echo -e "${GREEN}  ✓ All tests passed${NC}"
  else
    echo -e "${RED}  ✗ Some tests failed — check test_report.txt for details${NC}"
    if [[ "$TEST_ONLY" == "false" ]]; then
      echo -e "${YELLOW}  Continuing to screening despite test failures...${NC}"
    else
      exit 3
    fi
  fi

  if [[ "$TEST_ONLY" == "true" ]]; then
    echo ""
    echo -e "${GREEN}  Test-only mode complete.${NC}"
    exit 0
  fi
else
  echo ""
  echo -e "${BLUE}[3/5] Test suite${NC}"
  echo -e "  Skipped. Run with --test to verify system integrity first."
fi

# ── Step 4: Validate inputs ─────────────────────────────────────────────────
echo ""
echo -e "${BLUE}[4/5] Input validation${NC}"

if [[ "$SKIP_VAL" == "true" ]]; then
  echo -e "  Skipped (--skip-validation)"
else
  if python3 validate_inputs.py --ligands "$LIGANDS" --proteins "$PROTEINS"; then
    echo -e "${GREEN}  ✓ Validation passed${NC}"
  else
    VAL_EXIT=$?
    if [[ $VAL_EXIT -eq 1 ]]; then
      echo -e "${RED}  ✗ Validation errors found. Fix your input files before screening.${NC}"
      echo -e "     See validation_report.txt for details."
      exit 1
    fi
    # Exit 2 = file not found (already handled above)
  fi
fi

# ── Step 5: Run screening ──────────────────────────────────────────────────
echo ""
echo -e "${BLUE}[5/5] Running screening${NC}"
echo -e "  Model:      ${MODEL}"
echo -e "  Top-N:      ${TOP}"
echo -e "  Threshold:  ${THRESHOLD}"
echo -e "  Batch size: ${BATCH_SIZE}"
echo ""

SCREEN_ARGS="--model $MODEL --top $TOP --threshold $THRESHOLD --batch-size $BATCH_SIZE \
             --ligands $LIGANDS --proteins $PROTEINS --skip-validation"

if [[ "$NO_BINDING_SITES" == "true" ]]; then
  SCREEN_ARGS="$SCREEN_ARGS --no-binding-sites"
fi

if python3 screen.py $SCREEN_ARGS; then
  echo ""
  echo -e "${GREEN}${BOLD}=================================================================${NC}"
  echo -e "${GREEN}${BOLD}  Screening complete!${NC}"
  echo -e "${GREEN}${BOLD}=================================================================${NC}"

  # Find and display the output directory
  LATEST_OUTPUT=$(ls -td output/2*/ 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST_OUTPUT" ]]; then
    echo ""
    echo -e "  Output directory: ${BOLD}${LATEST_OUTPUT}${NC}"
    echo ""
    echo -e "  Files:"
    for f in results_full.csv results_top*.csv heatmap.csv binding_sites.csv run_summary.txt; do
      if [[ -f "${LATEST_OUTPUT}${f}" ]]; then
        SIZE=$(du -h "${LATEST_OUTPUT}${f}" | cut -f1)
        echo -e "${GREEN}    ✓ ${f}  (${SIZE})${NC}"
      fi
    done
    echo ""
    echo -e "  Quick preview — Top results:"
    if [[ -f "${LATEST_OUTPUT}results_full.csv" ]]; then
      head -6 "${LATEST_OUTPUT}results_full.csv" | column -t -s','
    fi
  fi
  exit 0
else
  echo -e "${RED}  ✗ Screening failed. Check the run_log.txt in the output directory.${NC}"
  exit 1
fi
