#!/usr/bin/env bash
# ============================================================================
# preprocess_teiban.sh -- clean a HUGE SMILES library on the Slurm cluster.
#
# De-salt, de-solvent, normalize and de-duplicate a big SMILES set (e.g. 700M),
# distributed across many CPU tasks. Charge / polarity is PRESERVED by default
# (add --neutralize to also uncharge to the neutral form).
#
#   bash preprocess_teiban.sh --input raw/ --output clean.smi --chunks 200 --maxpar 28
#   bash preprocess_teiban.sh --input smiles_001.txt --input smiles_002.txt \
#        --output clean.smi --chunks 400 --maxpar 28 --cpus 8
#
# This is CPU-only (no GPU): each array task runs predict_simple.py --preprocess
# on its chunk using all its cores, then a dependent job concatenates the chunk
# outputs and does a GLOBAL de-dup with `sort -u` (disk-based -> scales to 700M).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

OUTPUT=""; CHUNKS=100; MAXPAR=""; SIF="$HERE/teiban.sif"; DRYRUN=0; NEUTRALIZE=0
PARTITION="${TEIBAN_CPU_PARTITION:-${TEIBAN_PARTITION:-}}"; CPUS="${TEIBAN_CPUS:-8}"
TIME="${TEIBAN_TIME:-24:00:00}"; declare -a INPUTS=()

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -18; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --input) INPUTS+=("$2"); shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --chunks) CHUNKS="$2"; shift 2;;
    --maxpar) MAXPAR="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --sif) SIF="$2"; shift 2;;
    --neutralize) NEUTRALIZE=1; shift;;
    --dry-run) DRYRUN=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done
[ "${#INPUTS[@]}" -gt 0 ] || { echo "ERROR: --input is required (file or folder, repeatable)"; exit 1; }
[ -n "$OUTPUT" ] || { echo "ERROR: --output is required"; exit 1; }
if [ "$DRYRUN" = 0 ] && [ ! -f "$SIF" ]; then echo "ERROR: sif not found: $SIF"; exit 1; fi
SIF="$(readlink -f "$SIF")"
OUTDIR="$(cd "$(dirname "$OUTPUT")" 2>/dev/null && pwd || echo "$PWD")"; OUTPUT="$OUTDIR/$(basename "$OUTPUT")"

# Preprocessing is CPU-only -> use a CPU partition. If none given, take the Slurm
# default partition (the one marked '*'), which is almost always CPU.
if command -v sinfo >/dev/null 2>&1 && [ -z "$PARTITION" ]; then
  PARTITION="$(sinfo -h -o '%P' 2>/dev/null | awk '/\*/{gsub(/\*/,"");print;exit}')"
fi
[ -z "$PARTITION" ] && PARTITION="amd"

# Expand inputs (folders -> their SMILES files) into one list.
RAW_LIST=$(mktemp /tmp/teiban_prep_inputs.XXXX)
for it in "${INPUTS[@]}"; do
  if [ -d "$it" ]; then find "$it" -maxdepth 1 -type f \( -name '*.smi' -o -name '*.txt' -o -name '*.csv' -o -name '*.tsv' -o -name '*.ism' -o -name '*.smiles' \) >> "$RAW_LIST"
  elif [ -f "$it" ]; then echo "$it" >> "$RAW_LIST"; fi
done
[ -s "$RAW_LIST" ] || { echo "ERROR: no input SMILES files found"; exit 1; }
echo "[prep] input files: $(wc -l < "$RAW_LIST")   partition=$PARTITION (CPU)  cpus/task=$CPUS  chunks=$CHUNKS"

CDIR="$OUTDIR/teiban_prep_$$"; mkdir -p "$CDIR"
# Split into ~CHUNKS sequential files with `split -l` (opens ONE file at a time,
# so it never hits the open-file limit even for hundreds of chunks / 700M lines).
mapfile -t FILES < "$RAW_LIST"; rm -f "$RAW_LIST"
TOTAL=$(cat "${FILES[@]}" | wc -l)
[ "$TOTAL" -ge 1 ] || { echo "ERROR: no SMILES lines in input"; rm -rf "$CDIR"; exit 1; }
LINES=$(( (TOTAL + CHUNKS - 1) / CHUNKS )); [ "$LINES" -lt 1 ] && LINES=1
cat "${FILES[@]}" | split -l "$LINES" -d -a 4 --additional-suffix=.smi - "$CDIR/part_"
NPARTS=$(ls "$CDIR"/part_*.smi 2>/dev/null | wc -l)
[ "$NPARTS" -ge 1 ] || { echo "ERROR: split produced no chunks (empty input?)"; exit 1; }
echo "[prep] $TOTAL SMILES split into $NPARTS chunks (~$LINES each) -> $CDIR"

MAXTAG=""; if [ -n "$MAXPAR" ] && [ "$MAXPAR" -ge 1 ] 2>/dev/null; then MAXTAG="%$MAXPAR"; fi
NEUT=""; [ "$NEUTRALIZE" = 1 ] && NEUT="--neutralize"

ARR=$(mktemp /tmp/teiban_prep_arr.XXXX.sbatch)
cat > "$ARR" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=teiban_prep
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME
#SBATCH --array=0-$((NPARTS-1))$MAXTAG
#SBATCH --output=$CDIR/task_%a.log
set -e
P=\$(printf "%04d" \$SLURM_ARRAY_TASK_ID)
singularity exec "$SIF" python3 /opt/teiban/predict_simple.py --preprocess \\
    --drug-file "$CDIR/part_\$P.smi" --output "$CDIR/clean_\$P.smi" \\
    --workers \${SLURM_CPUS_PER_TASK:-$CPUS} $NEUT
EOF

MERGE=$(mktemp /tmp/teiban_prep_merge.XXXX.sbatch)
cat > "$MERGE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=teiban_prep_merge
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME
#SBATCH --output=$CDIR/merge.log
set -e
export LC_ALL=C
# global de-dup on the canonical SMILES (column 2); sort spills to disk -> scales.
cat $CDIR/clean_*.smi | sort -S 50% --parallel=$CPUS -t\$'\t' -k2,2 -u > "$OUTPUT"
echo "MERGED \$(ls $CDIR/clean_*.smi | wc -l) chunk(s) -> $OUTPUT  ( \$(wc -l < "$OUTPUT") unique molecules )"
EOF

if [ "$DRYRUN" = 1 ]; then
  echo "--- array sbatch ---"; cat "$ARR"; echo "--- merge sbatch ---"; cat "$MERGE"
  rm -rf "$CDIR"; exit 0
fi
JID=$(sbatch --parsable "$ARR")
echo "[prep] array job: $JID  ($NPARTS tasks$( [ -n "$MAXTAG" ] && echo ", up to $MAXPAR at once" ), $CPUS cpus each)"
sbatch --dependency=afterok:"$JID" "$MERGE"
echo "[prep] merge+dedup queued (runs after preprocessing) -> $OUTPUT"
echo "[prep] watch:  squeue -u \$(whoami)   |   tail -f $CDIR/merge.log"
