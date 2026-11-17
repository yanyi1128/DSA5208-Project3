# DSA5208 Project 3: 2D heat diffusion

C++ serial, CPU MPI and MPI+CUDA implementations of a double-precision five-point heat-diffusion stencil. The initial field is synthetic: a hot central square in an otherwise zero field, with fixed zero boundaries and diffusion coefficient `r = 0.1`. See [SPEC.md](SPEC.md) for the numerical, validation and CSV contracts.

## Implementation and measurement status

| Implementation | Available evidence | Scope and limitation |
| --- | --- | --- |
| Serial C++ | Correctness references and serial measurements described in the report | Keep the execution platform attached to each result. |
| CPU MPI | Vanda strong and weak scaling through 256 ranks; separate correctness experiments | Formal timing rows marked `not_run` were not individually validated. |
| MPI+CUDA | Actual local NVIDIA RTX 3070 Laptop GPU runs, one MPI rank and one GPU; `32x8` and `16x16` blocks | Local results are not Atlas/Vanda measurements. Vanda GPU execution and multi-rank/multi-GPU CUDA execution remain unmeasured. |

The Vanda CPU results are in [results/mpi_strong.csv](results/mpi_strong.csv) and [results/mpi_weak.csv](results/mpi_weak.csv), with concise provenance in [results/vanda_cpu_provenance.json](results/vanda_cpu_provenance.json). Local data are in [results/local_20261116](results/local_20261116/): `comparison_summary.csv`, `comparison_repetitions.csv` and `comparison_validation.csv` record the local serial/CUDA comparison; `block_summary.csv` and `block_repetitions.csv` record the local CUDA block-shape study; `provenance.json` identifies their sources and environment. A successful single-GPU test does not exercise inter-rank CUDA halo exchange. Empty legacy result files are placeholders, not measurements.

## Prerequisites and execution location

Run the commands below from the project root. Linux builds require a C++17 compiler; MPI builds also require an MPI development installation and launcher. CUDA builds require the CUDA Toolkit, an NVIDIA GPU and driver, and a CUDA-compatible host compiler. The Python CUDA tools require Python 3.8 or newer.

On Vanda, compile and execute only inside an allocated compute job. File transfer, job submission, status queries and `--dry-run` previews can be performed on the login node. Load modules appropriate to the allocated node and your authorised project. The supplied PBS templates are site-specific; they do not grant an allocation or repair account/project permissions.

The root `Makefile` is an empty legacy placeholder. Use the explicit commands below, including `-f Makefile.cuda` where shown. The empty legacy `scripts/run_serial.sh` is not a serial benchmark driver.

## Serial build and run

```bash
make -f Makefile.cuda serial
./heat_serial --n 64 --steps 100 --validate --output serial_64_100.csv
```

The `serial` target only needs the host C++ compiler; it does not require CUDA or MPI. An equivalent direct build is:

```bash
g++ -O2 -std=c++17 -Wall -Wextra -Wpedantic src/serial.cpp -o heat_serial
```

`--n` is the full square-grid width, including boundaries (`N >= 4`); `--steps` is the iteration count (`K >= 0`). The executable writes a timing/validation record to standard output. Optional `--output PATH` writes the final numerical field as an `N` by `N` CSV; its parent directory must already exist. Use small cases for validation and omit field export when collecting timing-only measurements.

## CPU MPI build, validation and scaling

On an allocated CPU compute node, after loading a compatible MPI/compiler environment:

```bash
mpicxx -std=c++17 -O2 -Wall -Wextra -Wpedantic src/mpi.cpp src/mpi_interface.cpp -o heat_mpi
mpirun -np 4 ./heat_mpi --n 64 --steps 100 --validate
```

This example requires at least four allocated CPU slots. The process count must not exceed either the allocation or `N`. The MPI executable supports `--n`, `--steps` and `--validate`; it does **not** implement `--output`. Its validation gathers the small final field and compares it with a serial reference on rank zero.

The scaling driver requires a real PBS allocation, includes one excluded warm-up followed by five measured runs by default, and rejects an existing output directory:

```bash
python3 scripts/benchmark_mpi.py --mode strong --n 64 --steps 100 \
  --ranks 1 2 4 --allocated-cores 4 --validate --output results/mpi_pilot_new
```

Add `--dry-run` to preview the commands without running anything. `--allocated-cores` must describe the resources actually granted; it is not a resource request. The final CPU experiments use `K = 1000`, strong-scaling `N = 4096`, and weak-scaling `N0 = 1024` with `N(P)` rounded from `N0 * sqrt(P)`, at ranks `1 2 4 8 16 32 64 128 256`. Do not substitute the older `N0 = 256` example for the measured weak-scaling configuration.

[docs/mpi-handoff.md](docs/mpi-handoff.md) documents the shared row-exchange interface. [docs/vanda-experiments.md](docs/vanda-experiments.md) and `scripts/submit_vanda_256.sh` describe the CPU submission workflow; statements in the former about allocations not yet obtained are historical preparation notes. Check current project access and node modules before reusing its submission commands.

## MPI+CUDA build and run on Vanda

The CUDA implementation uses one MPI rank per distinct allocated GPU and stages halo rows through host memory. It does not require CUDA-aware MPI. Link its main program with `src/mpi_interface.cpp`, not with the CPU executable's `src/mpi.cpp` main program.

On an allocated GPU compute node, with the CUDA Toolkit, compatible host compiler and Open MPI loaded:

```bash
make -f Makefile.cuda -j1 all serial
python3 scripts/test_cuda.py --exe ./heat_mpi_cuda --serial-exe ./heat_serial \
  --ranks 1 --allocated-gpus 1 --output results/cuda_validation_new
```

The validation script generates fresh serial reference fields, checks GPU fields against them, and performs independent analytic/boundary checks and expected-failure tests. Its output directory must be new. Only list rank counts supported by the distinct GPUs actually allocated. For example, `--ranks 1 2 --allocated-gpus 2` is a future two-GPU validation command, not evidence that two-GPU validation has passed.

For an unattended PBS job, submit from the project root:

```bash
qsub scripts/vanda_cuda.pbs
```

The default job runs a **pilot**: environment checks, a fresh source snapshot, compilation, and single-GPU validation with `32x8` blocks. It does not automatically collect the complete benchmark set. After the pilot succeeds and its results are inspected, submit the separate benchmark phase:

```bash
qsub -v CUDA_PHASE=benchmark scripts/vanda_cuda.pbs
```

The benchmark phase validates both block shapes and measures the configured single-GPU size/block comparisons. It creates a unique run directory under `results/runs`. Its summary analyses only the GPU measurements produced by that job; it does not automatically combine historical CPU data. Local and PBS snapshots need no historical timing CSVs, so the clean source package can run independently of the excluded legacy serial results. Optional CPU comparisons require separately selected inputs with verified platform and timing provenance. A queued job has not yet produced results; a running or completed job must still be checked for failures and validation outcomes. Do not submit a duplicate of a job that is already queued. Consult [docs/cuda-batch-guide.md](docs/cuda-batch-guide.md) for output/status inspection and [docs/cuda-guide.md](docs/cuda-guide.md) for manual build and multi-GPU experiment commands. The default module choices in `scripts/cuda_env_vanda.sh` still need verification on a Vanda GPU compute node.

## Local GPU reproduction

On Windows with an NVIDIA GPU and Docker Desktop's GPU-capable WSL2 Linux backend running:

```powershell
.\scripts\run_cuda_local.ps1 -Phase pilot
.\scripts\run_cuda_local.ps1 -Phase benchmark -SkipBuild
```

The first command builds the supplied Docker image. The second reuses that image, but the experiment controller still builds the solver from its fresh source snapshot. The default output root is `D:\DSA5208-CUDA-results`; override it with `-OutputRoot C:\path\to\results` if required. These runs are explicitly labelled **local**, not Atlas/Vanda. See [docs/cuda-local-guide.md](docs/cuda-local-guide.md) for prerequisites and limits.

For a configured local Linux CUDA/MPI environment:

```bash
python3 scripts/cuda_job.py --local --phase pilot
python3 scripts/cuda_job.py --local --phase benchmark
```

Use `--local` only for your own local GPU. Do not set a fake PBS job ID or use it to execute GPU work on a cluster login node.

After a successful local run, collect a matched serial/GPU comparison in that same CUDA/MPI environment, replacing `COMPLETED_LOCAL_RUN` with its actual run directory (containing `job_status.json` and both binaries):

```bash
python3 scripts/benchmark_local_comparison.py --validated-run COMPLETED_LOCAL_RUN \
  --output results/local_comparison_new
```

This driver runs a separate full-grid validation for both implementations at each of five sizes, one warm-up each, and five measured fresh processes, alternating execution order. It reuses the completed run's binaries and records their hashes. It defaults to `K=1000`, one rank/GPU and `32x8` blocks. It must run on a local GPU machine, and its output directory must be new. It is not run automatically by the pilot/benchmark wrapper.

To regenerate the three local report figures from the included curated CSVs, install Matplotlib in your analysis environment, then run:

```bash
python3 analysis/local_comparison_analysis.py --input results/local_20261116 \
  --output results/local_20261116/figures
```

## Host-side tests and reproducibility

```bash
python3 -m unittest discover -s tests -p 'test_cuda*.py' -v
make -f Makefile.cuda host-test
```

These test the CUDA tooling and host helpers without executing GPU kernels. They complement, and do not replace, the actual GPU validation above. CPU MPI interface tests are described in `docs/mpi-handoff.md`.

Retain source snapshots, exact commands, hardware/compiler metadata, raw repetitions and validation outcomes privately with each experiment. Keep local and cluster results distinguishable. Preserve `not_run` in timing records when validation was performed separately; do not relabel those rows as individually validated.

For the final submission, include source, this README, the report, and the selected CSV/figure results identified by the report. Exclude terminal transcripts and log files, as required by the marking guidance. Full `results/runs` directories are private evidence bundles, not final-submission folders. See [docs/submission-guide.md](docs/submission-guide.md) for packaging and team handoff.
