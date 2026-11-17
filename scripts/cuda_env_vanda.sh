#!/usr/bin/env bash
# Candidate modules seen in the user's Vanda module listing on 2026-11-14.
# Still requires verification of the GPU driver and compiler on a compute node.
# Source this file; module loading may use unset shell variables.
if ! type module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh || return 1
    else
        printf '%s\n' 'Environment Modules is unavailable in this shell.' >&2
        return 1
    fi
fi
module load CUDA/12.1.1 || return 1
module load OpenMPI/4.1.5-GCC-12.2.0 || return 1
export CUDA_HOST_CXX=g++
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
