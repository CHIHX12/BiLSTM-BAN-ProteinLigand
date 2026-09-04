# TEIBAN — debug handoff (for the Claude running on cluster 10.100.0.60)

You are on the GPU cluster `10.100.0.60` (user `cycheng`). Help finish getting the
TEIBAN drug–protein screening web tool to dispatch GPU jobs correctly on THIS
cluster. The image is already portable; this is almost certainly a small
cluster-specific config (partition name / gres / nested-sbatch), not a rebuild.

## Get the source (it is all inside the sif)
Everything runs from `/opt/teiban/` inside `teiban.sif` / `teiban-internal.sif`.
Extract it to look at / edit locally:
```bash
cd ~/teiban            # wherever the .sif is
singularity exec teiban-internal.sif cp -r /opt/teiban ~/teiban_src
ls ~/teiban_src        # teiban_web.py submit_teiban.sh preprocess_teiban.sh predict_simple.py PORTABILITY.md ...
```
The web server (`teiban_web.py`) is pure Python stdlib and runs on the LOGIN node.
Also read `~/teiban_src/PORTABILITY.md`.

## How it is supposed to work (architecture)
1. `teiban_web.py` (login node) submits ONE **controller** sbatch to a CPU
   partition (`--job-name=teiban_prep`). It does NOT do heavy work itself.
2. The controller (on a CPU node) builds the pairs CSV (`predict_simple.py
   --screen-csv`, uses rdkit inside the sif), splits it, then **nested-sbatch**es
   a GPU **array** job (`teiban_arr`) via `submit_teiban.sh` + a dependent merge.
3. The GPU array runs `singularity exec --nv teiban.sif predict.py` per chunk on
   the GPU nodes. Each task logs `nvidia-smi` + `Device: cuda`. Results merge into
   the output CSV. `afterany` + skip-done chunks make it resumable.

## THIS cluster (observed)
- GPU partition is **`gpu`**, nodes **`gnode1..gnode7`** (7 × 4 = 28 × V100 16GB).
- CPU partition is **`cpu`** (nodes `cnode*`). Slurm default partition may be `cpu`.
- A colleague (`chenyian`) already runs TEIBAN on `gpu`/gnode1, so the GPU path works.
- `cycheng`'s controller (`teiban_p` on `cpu`) runs, but we must confirm its GPU
  **array lands on `gpu`**.

## Symptom to fix
The user expects to see the job use the GPU nodes (`gnode1..7`) but so far only
saw the controller on `cpu`. Confirm the GPU array is dispatched to `gpu`, or make
it so.

## Diagnostics (run these)
```bash
sinfo -o "%P %a %G"                 # partition names + GRES; confirm 'gpu' has 'gpu:...'
sinfo -N -o "%N %P %G" | grep gnode # which partition gnode1..7 are in + gres string
scontrol show config | grep -i MaxArraySize
# the controller log (in the web Output folder the user chose):
cat "$(ls -t controller*.log 2>/dev/null | head -1)"   # look for '[submit] array job:' or an sbatch error
squeue -u cycheng
```

## Most likely fixes (in order)
1. **GPU partition not auto-detected.** `submit_teiban.sh` finds the GPU partition
   with `sinfo -h -o "%R %G" | awk 'tolower($0) ~ /gpu:/{print $1;exit}'`. If this
   cluster's GRES string is not `gpu:...` (e.g. blank, or `gres/gpu`), detection
   fails. Fix by forcing it, then restart the web:
   ```bash
   pkill -f teiban_web.py
   export TEIBAN_PARTITION=gpu     # the real GPU partition name here
   export TEIBAN_GRES=gpu          # or gpu:v100 if the site requires a type
   export TEIBAN_BATCH=64          # V100 16GB
   python3 teiban_web.py --sif "$PWD/teiban-internal.sif"
   ```
   (These env vars are read by submit_teiban.sh; the web passes --partition too.)
2. **Nested sbatch disabled** (a controller on a compute node cannot `sbatch`).
   Check the controller log for `sbatch: error`. If so, the fix is to change the
   web `submit()` in `teiban_web.py` so the login node submits the GPU array
   directly with `--dependency=afterok` on the (build-only) controller, instead of
   the controller nested-sbatching it. (Edit ~/teiban_src/teiban_web.py, run it
   from there; no sif rebuild needed for web changes.)
3. **Web partition dropdown empty** → `/api/cluster` didn't detect `gpu`. Same fix
   as (1): set `TEIBAN_PARTITION=gpu`.

## Config knobs (no rebuild needed)
`TEIBAN_PARTITION`, `TEIBAN_GRES`, `TEIBAN_CPUS`, `TEIBAN_BATCH`, `TEIBAN_TIME`,
and flags on `submit_teiban.sh` (`--partition/--gres/--gpus/--maxpar/--chunks/
--chunks-dir/--batch_size`). Web source: `teiban_web.py` (edit + rerun to iterate;
it is extracted from the sif, so changing the sif is only needed for handoff).

Repo (if internet): github.com/CHIHX12/BiLSTM-BAN-ProteinLigand
