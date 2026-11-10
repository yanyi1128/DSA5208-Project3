#!/usr/bin/env bash
# Call only after confirming your assigned project's CPU allocation.
set -euo pipefail
if [[ $# != 4 || ( "$2" != strong && "$2" != weak ) ]]; then
    echo "Usage: bash scripts/submit_vanda_256.sh PROJECT strong|weak N_OR_N0 STEPS" >&2
    exit 2
fi
project="$1"
export MODE="$2" GRID_N="$3" STEPS="$4"
if [[ ! "$GRID_N" =~ ^[1-9][0-9]*$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Grid and steps must be positive decimal integers' >&2
    exit 2
fi
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
export ALLOCATED_CORES=256 RANKS="1 2 4 8 16 32 64 128 256" VALIDATE=0
# Check the whole plan before consuming an allocation.
python3 scripts/benchmark_mpi.py --mode "$MODE" --n "$GRID_N" --steps "$STEPS" \
    --ranks 1 2 4 8 16 32 64 128 256 --allocated-cores 256 \
    --output results/preview --dry-run
# Four distinct hosts, 64 single-threaded ranks each. This is a request,
# not a claim that the scheduler has already granted these resources.
qsub -q auto -P "$project" -N "heat_${MODE}" \
    -l select=4:ncpus=64:mpiprocs=64:ompthreads=1:mem=16gb \
    -l place=scatter -l walltime=02:00:00 \
    -v MODE,GRID_N,STEPS,ALLOCATED_CORES,RANKS,VALIDATE \
    scripts/vanda_mpi.pbs
