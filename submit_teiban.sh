#!/usr/bin/env bash
# ============================================================================
# Submit a TEIBAN prediction to the Slurm cluster -- no hand-written sbatch.
#
#   bash submit_teiban.sh --input pairs.csv --output results.csv
#   bash submit_teiban.sh --input big.csv --output out.csv --chunks 8   # 8 GPUs
#
# Cluster note: GPU nodes live in partitions 'all' / 'intel' (2x RTX 6000 Ada
# each); the default 'amd' partition has NO GPU, so this script targets 'all'.
#
# --chunks N > 1 splits the input into N parts and runs a Slurm ARRAY (one GPU
# per part), then a dependent merge job concatenates the results into --output.
# This is how to use several GPUs at once (the model itself is single-GPU).
#
# DYNAMIC LOAD BALANCING: make N (--chunks) much LARGER than the number of GPUs
# and cap how many run at once with --maxpar G. Slurm then hands the next pending
# chunk to whichever GPU finishes first, so fast GPUs are never left idle:
#   bash submit_teiban.sh --input big.csv --output out.csv --chunks 200 --maxpar 8
#   # 200 small pieces, at most 8 on GPUs at a time, work-stealing across them.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

INPUT=""; OUTPUT=""; MODEL="BiLSTM"; PARTITION="all"; GPUS=1; CPUS=16
TIME="24:00:00"; CHUNKS=1; BATCH=128; SIF="$HERE/teiban.sif"; DRYRUN=0; MAXPAR=""

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --input) INPUT="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --chunks) CHUNKS="$2"; shift 2;;
    --maxpar) MAXPAR="$2"; shift 2;;
    --batch_size) BATCH="$2"; shift 2;;
    --sif) SIF="$2"; shift 2;;
    --dry-run) DRYRUN=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

[ -n "$INPUT" ]  || { echo "ERROR: --input is required"; exit 1; }
[ -n "$OUTPUT" ] || { echo "ERROR: --output is required"; exit 1; }
[ -f "$INPUT" ]  || { echo "ERROR: input not found: $INPUT"; exit 1; }
if [ "$DRYRUN" = 0 ] && [ ! -f "$SIF" ]; then echo "ERROR: sif not found: $SIF"; exit 1; fi
INPUT="$(readlink -f "$INPUT")"; SIF="$(readlink -f "$SIF")"
mkdir -p "$(dirname "$OUTPUT")" 2>/dev/null || true

run() { if [ "$DRYRUN" = 1 ]; then echo "[dry-run] would run: $*"; else "$@"; fi; }

# ---- single job ----------------------------------------------------------
if [ "$CHUNKS" -le 1 ]; then
  SBATCH=$(mktemp /tmp/teiban_job.XXXX.sbatch)
  cat > "$SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=teiban
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME
#SBATCH --output=teiban_%j.log
set -e
singularity exec --nv "$SIF" python3 /opt/teiban/predict.py \\
    --input "$INPUT" --output "$OUTPUT" \\
    --model $MODEL --batch_size $BATCH --num_workers $CPUS
echo "DONE -> $OUTPUT"
EOF
  echo "[submit] partition=$PARTITION gpus=$GPUS cpus=$CPUS  input=$INPUT"
  run sbatch "$SBATCH"
  [ "$DRYRUN" = 1 ] && { echo "--- generated sbatch ---"; cat "$SBATCH"; }
  exit 0
fi

# ---- chunked array (multi-GPU via data parallelism) ----------------------
CDIR="$(dirname "$(readlink -f "$OUTPUT")")/teiban_chunks_$$"
mkdir -p "$CDIR"
HEADER="$(head -1 "$INPUT")"
# round-robin rows into N chunk files, each with the header (order does not
# matter -- results are merged afterwards).
tail -n +2 "$INPUT" | awk -v n="$CHUNKS" -v dir="$CDIR" -v hdr="$HEADER" '
  { f = sprintf("%s/part_%03d.csv", dir, (NR-1) % n);
    if (!(f in seen)) { print hdr > f; seen[f] = 1 }
    print >> f }'
NPARTS=$(ls "$CDIR"/part_*.csv 2>/dev/null | wc -l)
[ "$NPARTS" -ge 1 ] || { echo "ERROR: split produced no chunks (empty input?)"; exit 1; }
echo "[submit] split into $NPARTS chunks -> $CDIR"

# Dynamic load balancing: cap concurrent array tasks at --maxpar (the GPUs to use
# at once). With NPARTS >> MAXPAR, Slurm assigns the next chunk to whichever GPU
# frees up first. No --maxpar -> all chunks eligible at once (bounded by the queue).
MAXTAG=""
if [ -n "$MAXPAR" ] && [ "$MAXPAR" -ge 1 ] 2>/dev/null; then MAXTAG="%$MAXPAR"; fi

ARRAY_SBATCH=$(mktemp /tmp/teiban_array.XXXX.sbatch)
cat > "$ARRAY_SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=teiban_arr
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME
#SBATCH --array=0-$((NPARTS-1))$MAXTAG
#SBATCH --output=$CDIR/task_%a.log
set -e
P=\$(printf "%03d" \$SLURM_ARRAY_TASK_ID)
singularity exec --nv "$SIF" python3 /opt/teiban/predict.py \\
    --input "$CDIR/part_\$P.csv" --output "$CDIR/pred_\$P.csv" \\
    --model $MODEL --batch_size $BATCH --num_workers $CPUS
EOF

MERGE_SBATCH=$(mktemp /tmp/teiban_merge.XXXX.sbatch)
cat > "$MERGE_SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=teiban_merge
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$CDIR/merge.log
set -e
first=1
for f in $CDIR/pred_*.csv; do
  if [ \$first = 1 ]; then cat "\$f" > "$OUTPUT"; first=0; else tail -n +2 "\$f" >> "$OUTPUT"; fi
done
echo "MERGED $NPARTS chunks -> $OUTPUT"
EOF

if [ "$DRYRUN" = 1 ]; then
  echo "--- array sbatch ---"; cat "$ARRAY_SBATCH"
  echo "--- merge sbatch ---"; cat "$MERGE_SBATCH"
  exit 0
fi
JID=$(sbatch --parsable "$ARRAY_SBATCH")
if [ -n "$MAXTAG" ]; then
  echo "[submit] array job: $JID  ($NPARTS tasks, up to $MAXPAR on GPUs at once -- dynamic)"
else
  echo "[submit] array job: $JID  ($NPARTS tasks, 1 GPU each)"
fi
sbatch --dependency=afterok:"$JID" "$MERGE_SBATCH"
echo "[submit] merge job queued (runs after the array finishes) -> $OUTPUT"
