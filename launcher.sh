#!/bin/sh

app_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Keep the packaged application responsive while PPO updates run in its worker.
# An explicitly supplied value, including an intentionally empty one, is left alone.
if [ "${JULIA_NUM_THREADS+x}" != x ]; then
    JULIA_NUM_THREADS=2
fi
export JULIA_NUM_THREADS

# OpenBLAS otherwise inherits machine-wide thread counts for tiny policy matrices.
# Respect any caller-provided value, even an intentionally empty one.
if [ "${OPENBLAS_NUM_THREADS+x}" != x ]; then
    OPENBLAS_NUM_THREADS=1
fi
export OPENBLAS_NUM_THREADS

export LD_LIBRARY_PATH="$app_dir/lib/julia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$app_dir/bin/War1gusAI.bin" "$@"
