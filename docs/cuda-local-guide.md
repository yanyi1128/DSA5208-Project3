# Local CUDA fallback and Vanda experiments

Prepared 16 November 2026. Local measurements must be labelled as local; the assignment explicitly requests Atlas/Vanda runtime figures. Existing group CPU MPI results from Vanda remain separate evidence. Obtain the instructor's direction if C cannot obtain Vanda GPU measurements.

## Local Windows entry point

Observed hardware: NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB VRAM, Windows driver 591.74. The container uses Ubuntu 22.04, CUDA 12.1 compiler/runtime headers and Open MPI. It runs as an unprivileged user with one GPU, at most four CPU cores and 6 GiB RAM. The default CUDA architecture is `sm_86`, appropriate to this GPU; edit it for another architecture.

From the project directory in PowerShell:

```powershell
.\scripts\run_cuda_local.ps1 -DryRun
.\scripts\run_cuda_local.ps1 -Phase pilot
.\scripts\run_cuda_local.ps1 -Phase benchmark -SkipBuild
```

The first real command builds `scripts/docker-local/Dockerfile` using the official NVIDIA base image, then compiles and validates the project. The benchmark phase additionally validates the other block shape and measures N=256,512,1024,2048,4096, K=1000, for blocks 32x8 and 16x16. It retains one excluded warm-up and five measured repetitions per configuration. Run the benchmark only after pilot succeeds.

Results go to unique directories under `D:\DSA5208-CUDA-results` by default. Change this with `-OutputRoot` if needed. Docker image storage on this machine is already on E:, at `E:\DockerData\DockerDesktopWSL`. The project is mounted read-only; the run freezes a source snapshot and writes binaries, logs and results in its own result directory. No Vanda job is submitted or cancelled by this script.

Every run saves a host provenance JSON, `job_status.json`, environment/command logs, source hashes and validation results. `status=complete` must be supported by the stage records. Failed runs retain their logs and must not be described as passed. The program's ordinary timing records use `validation=not_run` when the individual measured run did not perform a full-field check; separate correctness tests remain separate evidence.

## Linux entry point

With nvcc, g++, make, Python >=3.8 and Open MPI available:

```bash
python3 scripts/cuda_job.py --local --phase pilot
python3 scripts/cuda_job.py --local --phase benchmark
```

The default Linux local output is `results/local_runs/cuda_local_<unique suffix>`. The `--local` flag is propagated to validation and benchmarking, records no PBS job and refuses to run if `PBS_JOBID` is also set. Do not fabricate PBS variables. The local summary deliberately omits the group's Vanda CPU baselines; a within-machine GPU/CPU speedup needs a locally measured serial baseline.

## Vanda path

The existing PBS template and default PBS requirement are preserved. The cluster directory is `~/project3_cuda/DSA5208-Project3-main`; these local source changes have not automatically been uploaded there.

```bash
qstat -fx 1555991.stdct-mgmt-02
qstat -T 1555991.stdct-mgmt-02
```

The latest supplied snapshot (16 November, 16:02:45) is Q with `Insufficient amount of resource: node_pool`, no hold, no estimated start time. An earlier snapshot reported `personal-gujiaxuan` missing. A changed scheduler comment does not confirm the accounting/project mapping is fully repaired. Keep the existing batch job and update the original IT support ticket. Once allocated, the batch script starts automatically; the default pilot validates but is not the full benchmark.

## What a local result can establish

- CUDA compilation and single-GPU correctness on this specific local environment.
- Local problem-size and block-shape performance, when actually measured.
- It cannot establish multiple-GPU halo correctness, multi-GPU scaling, or Vanda performance.

The preparation checks passed 23 Python controller/analysis tests. Actual local GPU execution subsequently completed:

- Pilot: `local_pilot_20261116_091046_5fc69fe6`.
- Benchmark: `local_benchmark_20261116_091130_7a29270f`, complete; both blocks passed ten positive and six negative tests. Across these suites, maximum absolute error was 5.551115123125783e-17 and maximum relative L2 error was 1.4901003784504166e-17.
- Ten timing configurations completed with five measured repetitions plus one excluded warm-up each. At N=4096/K=1000, median 0.662353475 s for 32x8 and 0.699719365 s for 16x16. This is a local block comparison, not multi-GPU scaling or CPU/GPU speedup. The sweeps were sequential and laptop thermal/power conditions were not controlled.
- Actual toolchain: CUDA 12.1.105, GCC 11.4.0, Open MPI 4.1.2, Python 3.10.12. Source and binary hashes are in the retained runs.

Docker initially stopped after an `ENOSPC` error in its frontend log; restarting restored the engine. No user data or other images/containers were deleted. If it fails again, check disk space and Docker's logs before retrying. A successful driver-level `nvidia-smi` alone does not prove the CUDA application works.

Reference: [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/) and [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
