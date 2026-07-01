# Running teiban as a portable container (.sif)

This packages the entire `drugban` environment (Python 3.10 + PyTorch 2.2.1/cu121
+ DGL 2.1.0/cu121 + RDKit + the trained checkpoints + all prediction code) into a
**single `teiban.sif` file**. Copy that one file to any server with an NVIDIA GPU
and Apptainer/Singularity — no conda, no pip, no setup, no environment problems.

## Files

| File | Purpose |
|------|---------|
| `teiban.def` | Singularity/Apptainer build recipe (GPU, CUDA 12.1) |
| `build_sif.sh` | Builds `teiban.sif` rootless, with scratch on `/home` |
| `Dockerfile` | Alternative: build a Docker image (for hosts with disk room) |
| `.dockerignore` | Keeps the Docker image lean |

## Build the image (on THIS server — no GPU needed to build)

```bash
bash build_sif.sh
# produces ./teiban.sif  (~5 GB)
```

The script redirects Singularity's cache + temp to `$HOME` (on `/home`, which has
free space) because `/` on this machine is ~98% full. It builds rootless via
`--fakeroot`, so no `sudo` is required.

## Run it (on a server WITH an NVIDIA GPU)

Copy `teiban.sif` to the GPU server, then:

```bash
# GPU self-check + help banner
singularity run --nv teiban.sif

# Simple predictor -- run with NO args for a guided number menu:
#   (1) type one drug + one protein   (2) load a folder with smiles.txt + protein_seq.txt
#   then pick model: (1) BiLSTM  (2) CNN  (3) Both (compare side by side)
singularity exec --nv teiban.sif python3 /opt/teiban/predict_simple.py

# Or drive it directly (no menu). Input = file OR folder of pairs; cols (header
# optional): [name,] SMILES, Protein.  --model both compares BiLSTM vs CNN.
singularity exec --nv teiban.sif python3 /opt/teiban/predict_simple.py \
    --input pairs.csv --output results/ --model both
# Single pair:
singularity exec --nv teiban.sif python3 /opt/teiban/predict_simple.py \
    --drug "CC(=O)Oc1ccccc1C(=O)O" --protein MENFQKVEK...
# If --output is omitted, the CSV is saved next to teiban.sif (else current dir).

# Batch / N:M screening -- every drug x every protein (see bundled examples)
singularity exec --nv teiban.sif python3 /opt/teiban/predict_batch.py \
    --ligands /opt/teiban/examples/drugs.txt \
    --receptors /opt/teiban/examples/proteins.fasta \
    --output out.csv

# CLI prediction (single pair, CSV, or screen modes)
singularity exec --nv teiban.sif python3 /opt/teiban/predict.py --help
```

- `--nv` exposes the host GPU. **Without** `--nv` (or on a CPU-only host) the code
  automatically falls back to CPU — slower, but it still runs.
- The GPU host only needs an NVIDIA **driver** new enough for CUDA 12.1
  (driver >= 530). The CUDA runtime itself is inside the image.

## Reading/writing your own files

The container filesystem is read-only. By default Singularity bind-mounts your
`$HOME` and current directory, so files there are visible. To expose other paths
or write outputs explicitly:

```bash
# Mount an input/output folder to /data inside the container
singularity exec --nv -B /path/on/host:/data teiban.sif \
    python3 /opt/teiban/predict.py --input /data/pairs.csv --output /data/out.csv
```

## What's inside vs. what's not

- **Inside** `/opt/teiban`: all `*.py` code, `configs/`, trained checkpoints in
  `result/` (BiLSTM `epoch_94`, CNN `epoch_90`), `examples/`, `P-L/`, screening code.
- **Not bundled** (to keep the image portable): `FDA_drugs/`, `datasets/`,
  `fda_validation/`, generated outputs. Bind-mount them with `-B` if a script needs them, e.g.:

  ```bash
  singularity exec --nv -B $PWD/datasets:/opt/teiban/datasets teiban.sif ...
  ```

## Rebuilding after code changes

Re-run `bash build_sif.sh`. The base image and Python wheels are cached in
`$HOME/.singularity/cache`, so subsequent builds are much faster.

## Docker route (only if a host has disk room for Docker)

```bash
docker build -t teiban:gpu .
docker run --rm -it --gpus all teiban:gpu                       # self-check
docker run --rm -it --gpus all teiban:gpu python3 /opt/teiban/predict_simple.py
# Convert a Docker image to .sif:
singularity build teiban.sif docker-daemon://teiban:gpu
```
