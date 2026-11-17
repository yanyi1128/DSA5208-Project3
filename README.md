
# DSA5208 Project 3: Parallelizing 2D Heat Diffusion Using MPI and CUDA

## 1. Project Overview

This project implements a two-dimensional heat diffusion simulation using three approaches:

- Serial C++
- MPI for distributed-memory CPU parallelism
- MPI+CUDA for GPU acceleration

The numerical method uses an explicit five-point finite-difference stencil with double-precision values. The initial condition consists of a hot square in the center of the domain, with fixed zero-temperature boundaries. The diffusion coefficient used in the update is r = 0.1.

The project compares correctness, execution time, and parallel performance across the implementations.

## 2. Repository Structure

- `src/` — Serial, MPI, and MPI+CUDA source files
- `include/` — Shared headers and interfaces
- `tests/` — Correctness and validation tests
- `scripts/` — Benchmarking, validation, and cluster job scripts
- `results/` — Measurement data and performance figures
- `docs/` — Additional compilation and experiment instructions
- `Makefile.cuda` — Build configuration for Serial and MPI+CUDA
- `SPEC.md` — Numerical algorithm and validation specifications

## 3. Compilation

The programs are intended to be compiled in a Linux environment. The MPI implementation requires an MPI compiler, while the CUDA implementation also requires the NVIDIA CUDA Toolkit and a compatible GPU.

### Serial C++

```bash
make -f Makefile.cuda serial
```

This produces the `heat_serial` executable.

### MPI

```bash
mpicxx -O2 -std=c++17 -Iinclude \
  src/mpi.cpp src/mpi_interface.cpp \
  -o heat_mpi
```

This produces the `heat_mpi` executable.

### MPI+CUDA

```bash
make -f Makefile.cuda
```

This produces the `heat_mpi_cuda` executable.

The CUDA build uses `nvcc` and an MPI compiler. Environment-specific settings, including the CUDA architecture, can be configured through `Makefile.cuda`.

## 4. Running and Validation

### Serial

```bash
./heat_serial 64 100 --validate
```

### MPI

```bash
mpirun -np 4 ./heat_mpi 64 100 --validate
```

### MPI+CUDA

Compile the CUDA implementation on a machine with a compatible NVIDIA GPU and an MPI installation. The CUDA benchmark and validation tools are available through:

```bash
python3 scripts/test_cuda.py --help
python3 scripts/benchmark_cuda.py --help
```

Additional environment-specific instructions are provided in `docs/cuda-guide.md` and `docs/cuda-local-guide.md`.

The implementations are validated against reference numerical results using maximum absolute error and relative L2 error. The tests also check boundary conditions and numerical consistency.

## 5. Performance Experiments

Serial and CPU MPI experiments were conducted on the NUS Vanda HPC cluster. MPI performance was evaluated using strong and weak scaling with up to 256 CPU ranks.

CUDA correctness and performance experiments were conducted on a local NVIDIA RTX 3070 Laptop GPU, including comparisons between CUDA thread-block configurations and matched serial/GPU execution times.

Measurement data and figures are available in:

- `results/` — CPU performance results
- `results/local_20261116/` — Local CUDA and serial comparison results

The report provides the experimental methodology, timing results, speedup calculations, and performance analysis.

## 6. Additional Documentation

For further information, see:

- `SPEC.md` — Numerical and validation specifications
- `docs/vanda-experiments.md` — Vanda CPU experiments
- `docs/cuda-guide.md` — CUDA compilation and execution
- `docs/cuda-local-guide.md` — Local CUDA environment
- `docs/submission-guide.md` — Submission and reproducibility guidance

