# Moving TEIBAN to another cluster

The `teiban.sif` / `teiban-internal.sif` image is self-contained and does **not**
need rebuilding for a different cluster or a different GPU (e.g. V100). Only a few
site-specific things differ. Most are auto-detected; the rest are one-line env vars.

## 1. Copy the image to shared storage
Put the `.sif` where every compute node can read it (usually `/home` or a shared
`/scratch`). Jobs run `singularity exec --nv <sif> ...` on the GPU nodes, so the
path must resolve there too.

## 2. Check the GPU driver (the one real compatibility gate)
The image ships PyTorch 2.2.1 + CUDA **12.1** and DGL 2.1.0 (cu121). The GPU
architecture is fine (V100 = sm_70 is supported), but the **host driver must
support CUDA 12.1** — i.e. NVIDIA driver **>= 525**. Check on a GPU node:

```bash
srun --gres=gpu:1 nvidia-smi          # look at "Driver Version" and GPU memory
```

Each TEIBAN job also prints the GPU name + driver + memory at the top of its log,
and `predict.py` prints `Device: cuda` (good) or `Device: cpu` (driver too old /
no GPU visible — it will run, but slowly). If the driver is < 525 you need a sif
built against an older CUDA (cu118); ask the maintainer.

## 3. Partition name — auto-detected
`all` / `intel` are specific to the origin cluster. `submit_teiban.sh` runs
`sinfo` and automatically uses the first partition that has a GPU gres, so you
usually do nothing. To force one:

```bash
export TEIBAN_PARTITION=gpu        # whatever your GPU partition is called
```

The web UI's Partition dropdown is also populated from `sinfo` automatically.

## 4. GPU memory — batch size (V100 16 GB vs RTX 6000 Ada 48 GB)
The default batch size (128) is sized for 48 GB cards. On 16 GB V100s, lower it to
avoid out-of-memory:

```bash
export TEIBAN_BATCH=64             # or 32 if you still hit OOM
```

In the web UI there is a **Batch size** field (hint: 64 for 16 GB).

## 5. Other one-line overrides (only if needed)
```bash
export TEIBAN_GRES=gpu:v100        # if the site requires a typed gres
export TEIBAN_CPUS=8               # cpus per task (fewer cores per GPU)
export TEIBAN_TIME=12:00:00        # if the queue caps wall time
```

`--gres / --cpus / --time / --batch_size / --partition` also work as flags on
`submit_teiban.sh`.

## 6. GPU count is irrelevant to correctness
28 GPUs vs 16 — no change needed. Submissions split the input into many small
pieces and run a Slurm array capped at the GPUs you ask for (`--maxpar`), so work
is handed to whichever GPU frees up first. Just pick the number of GPUs you want.

---

**Quick start on a new cluster (V100 example):**
```bash
export TEIBAN_BATCH=64                       # 16 GB cards
# partition + gres auto-detected; set TEIBAN_PARTITION only if detection is wrong
singularity exec teiban-internal.sif cat /opt/teiban/teiban_web.py > teiban_web.py
python3 teiban_web.py                         # open the printed URL
```
