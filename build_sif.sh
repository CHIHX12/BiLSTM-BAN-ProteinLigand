#!/usr/bin/env bash
# ============================================================================
# Build the teiban GPU .sif image with SingularityCE.
#
# Why this script: on this machine /var/lib/docker and /tmp live on "/", which
# is ~98% full. Singularity's cache + build temp are redirected to $HOME
# (on /home, which has plenty of free space) so the build does not fail on
# "no space left on device". No GPU and no root are required to BUILD.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# Keep all heavy scratch on /home (huge), never on the near-full / filesystem.
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$HOME/.singularity/tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.singularity/cache}"
# Apptainer-named vars too, in case the binary reads those.
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

OUT="${1:-teiban.sif}"

echo "[build_sif] TMPDIR=$SINGULARITY_TMPDIR"
echo "[build_sif] CACHEDIR=$SINGULARITY_CACHEDIR"
echo "[build_sif] Output  =$OUT"
echo "[build_sif] Building (rootless, --fakeroot). This downloads ~3-5 GB and takes a while..."

singularity build --fakeroot "$OUT" teiban.def

echo "[build_sif] Done. Image: $(readlink -f "$OUT")"
ls -lh "$OUT"
