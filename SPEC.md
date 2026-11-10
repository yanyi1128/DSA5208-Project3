# NUS DSA5208 Project 3 — 2D Heat Diffusion

Implementation and integration specification for Serial, MPI, and MPI+CUDA.

## 1. Status, authority, and requirement labels

This document is a **team numerical and implementation contract**. Concrete MPI function signatures will be documented after the MPI implementation is complete. It specifies one consistent numerical problem and comparable implementations.

**Source limitation:** The team division-of-work PDF supplies the original experiment plan, including the 256-CPU-core MPI experiment. The official assignment handout has not been independently verified here.

| Label | Meaning |
|---|---|
| COURSE-SCOPE (user-provided; unverified against handout) | NUS DSA5208 Project 3; 2D heat diffusion; Serial, MPI, and MPI+CUDA implementations. |
| TEAM-DESIGN | A binding choice for this team's implementation, including the requested numerical settings. It is not presented as an instructor-mandated requirement. |
| COURSE-CHECK | An item the team must verify against the official assignment before submission. |

**All normative rules below are TEAM-DESIGN unless explicitly labeled otherwise.** “Must” means required for team interoperability. If the official handout conflicts with this document, update the specification, all implementations, and the experiment records together. Do not silently change one implementation.

### COURSE-CHECK: submission gate

Confirm required languages, build tools, cluster and GPU environment, process/GPU restrictions, prescribed problem sizes and iteration counts, required experiments and assessment criteria against the official handout. Confirm whether any particular boundary conditions, initialization, accuracy thresholds, or timing definitions are prescribed. This list records unknowns, not additional course requirements.

## 2. Shared numerical problem

### 2.1 Grid, indexing, and parameters

- The global grid has **N × N stored points, including boundary points**.
- Global indices are zero-based: row `i` and column `j`, each in `[0, N)`.
- Require `N >= 4` and a nonnegative integer iteration count `K`.
- All field values and stencil arithmetic use IEEE 754 `double` (64-bit floating point).
- The fixed diffusion coefficient per time step is **r = 0.1** in both spatial directions.
- For the standard explicit discretization, `r = alpha * dt / h^2` with equal spacing in both directions. The implementation directly uses `r`; no physical units or separate `alpha`, `dt`, or `h` are needed.
- The square-grid five-point scheme is stable for `0 <= r <= 1/4`; the chosen `r` satisfies this condition.
- Execute exactly `K` updates. Do not stop early based on convergence.

### 2.2 Initial condition

Define integer bounds using integer arithmetic:

```text
lo = floor(N / 4)
hi = floor(3 * N / 4)
```

Compute `3 * N` in a sufficiently wide integer type. In particular, **do not replace `hi` with `3 * floor(N / 4)`**; these differ when `N` is not divisible by four.

At time step zero:

```text
u[i,j] = 1.0  if lo <= i < hi and lo <= j < hi
u[i,j] = 0.0  otherwise
```

The central hot square is therefore the half-open region **`[N/4, 3N/4) × [N/4, 3N/4)`**, interpreted with the integer bounds above. Initialize every physical boundary to zero explicitly.

**Temperature normalization:** All three implementations store and compute the dimensionless normalized temperature `u`, not temperature in degrees Celsius. Using the original cold and hot temperatures `T_cold = 0 degrees C` and `T_hot = 100 degrees C`, define:

```text
u = (T - T_cold) / (T_hot - T_cold)
T = T_cold + u * (T_hot - T_cold)
```

Thus `u=1.0` represents 100 degrees C, `u=0.0` represents 0 degrees C, and `u=0.5` represents 50 degrees C. The linear heat-diffusion update has the same form under this normalization; keep `r=0.1` unchanged. Initialization, stencil computation, validation thresholds, and output field CSV values all use normalized `u`. Do not mix Celsius values with normalized values across implementations. Any optional visualization converted to Celsius must explicitly label the conversion and units.

### 2.3 Fixed physical boundaries

For every time step, including zero:

```text
u[0,j] = u[N-1,j] = u[i,0] = u[i,N-1] = 0.0
```

These are homogeneous Dirichlet boundaries. They are neither periodic nor insulated. Physical boundary points must never be updated by the stencil. A subdomain boundary between MPI ranks is not a physical boundary.

### 2.4 Update rule

For all global interior points `1 <= i,j <= N-2`:

```text
next[i,j] = current[i,j] + 0.1 * (
    current[i-1,j] + current[i+1,j]
  + current[i,j-1] + current[i,j+1]
  - 4.0 * current[i,j]
)
```

All reads in an iteration come from the same old time level. **In-place updates are forbidden.** Keep the written expression and operand order consistent where practical; compiler and GPU arithmetic may still differ slightly, so correctness is tolerance-based rather than bitwise.

## 3. Storage and buffer ownership

- Store each field contiguously in row-major order: `offset(i,j) = i * N + j`.
- Use wide unsigned indexing (for example `size_t`) for element offsets and checked size multiplication before allocation.
- Maintain two equally sized arrays named conceptually `current` and `next`.
- Serial arrays each contain `N * N` doubles.
- MPI and MPI+CUDA arrays each contain `(local_rows + 2) * N` doubles, including one halo row on each side.
- Zero both arrays initially, then populate the initial condition in the owned rows of `current`.
- Physical boundary cells remain zero in both arrays for their entire lifetime.
- After each completed update, swap the two buffer handles. After `K` swaps, **`current` is the final result**, including when `K = 0` or `K` is odd.
- Neither implementation may assume that a particular original allocation always holds the final result.

`SPEC.md` documents the contract; it does not configure the executable automatically. Implement shared defaults, constants, argument validation, and decomposition logic in source code. Change this document and the relevant source together whenever the contract changes.

## 4. MPI row decomposition

### 4.1 Owned rows and uneven partitions

Use a one-dimensional decomposition of **all N global rows, including the physical top and bottom boundary rows**. Let `P` be the MPI communicator size and `p` the rank in `[0,P)`.

Require `1 <= P <= N`. Reject `P > N` collectively with a clear error; zero-owned-row ranks are not supported in this contract.

```text
q = N / P                 # integer division
s = N % P
local_rows(p) = q + (p < s ? 1 : 0)
start_row(p)  = p*q + min(p,s)
end_row(p)    = start_row(p) + local_rows(p)   # exclusive
```

Each rank owns the consecutive global rows `[start_row, end_row)`. Local layout:

```text
local row 0                  = top halo
local rows 1..local_rows     = owned rows
local row local_rows + 1     = bottom halo
global_row(local_row)        = start_row + local_row - 1
local_offset(local_row,j)    = local_row * N + j
```

Rank `p-1` is the upper neighbor and rank `p+1` is the lower neighbor. Use `MPI_PROC_NULL` where a neighbor does not exist. Initialize each owned cell directly from its global coordinates; no initial scatter is needed.

Example: `N=10, P=3` gives row counts `[4,3,3]` and starts `[0,4,7]`.

### 4.2 Halo exchange contract

Exchange complete contiguous rows of **N `MPI_DOUBLE` values**, including the zero left/right endpoints, from `current` at the beginning of every iteration. Use `MPI_COMM_WORLD` or one agreed equivalent communicator consistently.

Define two distinct tags shared by both parallel implementations:

```text
TAG_TO_UP   = 100
TAG_TO_DOWN = 101
```

The required baseline communication order is:

```text
Irecv(current row 0,              from upper, tag TAG_TO_DOWN)
Irecv(current row local_rows+1,   from lower, tag TAG_TO_UP)
Isend(current row 1,              to upper,   tag TAG_TO_UP)
Isend(current row local_rows,     to lower,   tag TAG_TO_DOWN)
Waitall(all four requests)
compute all owned, globally interior points into next
swap(current, next)
```

Post receives before sends. Do not read received halos before completion, and do not modify or reuse an active send buffer before its send completes. Reuse the same tags each iteration only because all requests complete before the next iteration begins. There is no per-iteration global barrier.

`MPI_PROC_NULL` receives leave their destinations unchanged; initialize those halo rows to zero. They are outside the physical domain and are not needed when physical boundary rows are correctly skipped.

The algorithm must handle `local_rows = 1` and `local_rows = 2`. A one-row rank may send the same read-only row to both neighbors using separate requests and tags. Skip any owned row whose global index is `0` or `N-1`.

### 4.3 Optional communication overlap

The baseline above is the required integration path. An optimized variant may compute local rows `2..local_rows-1` while requests are active, then wait and compute edge rows `1` and `local_rows`, skipping global physical boundaries and avoiding duplicate updates when there is only one owned row. Document and time the optimized variant separately. Preserve buffer lifetimes and numerical semantics.

### 4.4 Final reconstruction

For validation or output, gather owned rows only:

```text
send address = current + N
send count   = local_rows(p) * N
counts[p]    = local_rows(p) * N
displs[p]    = start_row(p) * N
```

Use `MPI_Gatherv` with `MPI_DOUBLE` to reconstruct row-major global storage on rank zero. Never gather halo rows. Check counts and displacements against the actual MPI API integer limits; reject unsupported sizes or implement a documented large-count/chunked path. Do not silently truncate.

## 5. Executables and command-line contract

Provide three executables with matching numerical options:

```text
heat_serial   --n N --steps K [--validate] [--output PATH]
heat_mpi      --n N --steps K [--validate] [--output PATH]
heat_mpi_cuda --n N --steps K [--validate] [--output PATH]
```

- `--n` and `--steps` are required. No undocumented defaults.
- `--help` prints usage and exits successfully without running a simulation.
- `r=0.1`, initial hot value `1.0`, and boundary value `0.0` are fixed constants, not runtime options.
- `--validate` runs the correctness procedure in Section 7, outside timing.
- `--output PATH` writes the final global field, outside timing. Only serial or MPI rank zero writes the output file.
- Unknown options, malformed integers, invalid dimensions, overflow, allocation failures, CUDA failures, and unsupported process counts produce a nonzero exit status. Parallel failures must terminate coherently rather than strand peers in communication.
- Rank zero parses and broadcasts the validated configuration and parse status; all ranks take the same execution or exit path.

Example commands:

```sh
./heat_serial --n 1024 --steps 1000
mpirun -np 4 ./heat_mpi --n 1024 --steps 1000
mpirun -np 4 ./heat_mpi_cuda --n 1024 --steps 1000
mpirun -np 3 ./heat_mpi --n 17 --steps 7 --validate --output final.csv
```

These are interface examples, not guaranteed cluster launch commands. Adapt the launcher and resource allocation to the actual environment.

Output field format: UTF-8 CSV with exactly N rows and N comma-separated values per row, no header, global row order, and at least 17 significant decimal digits per value. Reject an unwritable output path with an explicit error.

Print one machine-readable CSV result record on rank zero, with a header:

```text
implementation,n,steps,r,ranks,gpus,elapsed_s,validation,max_abs_error,relative_l2_error
```

Use implementation labels `serial`, `mpi`, and `mpi_cuda`; GPU count is zero for CPU executables. Set validation to `not_run`, `pass`, or `fail`; use `NA` for uncomputed error metrics. Send descriptive diagnostics to stderr. A failed requested validation returns nonzero on every rank.

## 6. CUDA integration contract

### 6.1 Responsibilities and mapping

- MPI owns decomposition, global configuration, neighbor selection, message tags, timing reduction, final gathering, and reporting.
- CUDA owns two device fields with exactly the local layout in Section 4, stencil launches, synchronization, and device/host copies.
- The baseline uses **one MPI rank per GPU with exclusive GPU assignment**. No GPU sharing or multi-GPU control by a single rank is required.
- Determine node-local rank with a shared-memory MPI communicator. If the scheduler exposes one assigned device per rank, select visible device `0`. If ranks see the same full device list, select the node-local rank's device. Reject oversubscription and record the chosen mapping; do not assume global MPI rank is a device ordinal.

### 6.2 Required portable path: host-staged MPI

Do not assume CUDA-aware MPI. Allocate separate host send/receive row buffers, preferably pinned. Initialize host receive buffers to zero so absent-neighbor operations are safe.

Each iteration must execute in this order:

1. Ensure previous writes to device `current` have completed.
2. Copy device owned row `1` and row `local_rows` to the two host send buffers. Complete these copies before passing the buffers to MPI.
3. Post the two host-buffer receives and then the two host-buffer sends using Section 4.2's neighbors and tags.
4. Complete all four MPI requests before reusing host buffers or consuming received data.
5. Copy received rows into device halo rows `0` and `local_rows+1` for real neighbors. Preserve zero exterior halos for absent neighbors.
6. Complete halo copies before any dependent stencil kernel begins, using one ordered CUDA stream or explicit events and waits.
7. Launch the stencil kernel, writing owned globally interior points into device `next`.
8. Swap device buffer handles. Ensure the next iteration's copies are ordered after this kernel. Synchronize at the end of the timed loop before stopping the timer.

Keep fields resident on the GPU across iterations. Only the boundary-row staging traffic occurs each iteration; a whole-field round trip each iteration is prohibited. Final device-to-host copying occurs after timing.

CUDA errors must be checked after allocations, transfers, launches, and synchronizations. A local failure must trigger coordinated failure of the MPI job, for example via `MPI_Abort`.

### 6.3 Kernel and API boundary

Use a two-dimensional launch covering columns `j=1..N-2` and owned local rows `ell=1..local_rows`; map each thread to one cell. Guard excess threads and skip global rows `0` and `N-1`. Use `ell*N+j` indexing and the same stencil as the CPU. Block shape, such as `32 × 8`, is a tunable team choice, not a course requirement. Use ordinary contiguous device allocation in the baseline; pitched storage would require a coordinated contract change.

Agree on an interface equivalent to:

```cpp
struct Domain {
    int n;
    int local_rows;
    int start_row;
};

// These are semantic interfaces; concrete CUDA types live in the CUDA header.
GpuState* gpu_create(Domain domain, const double* host_current_with_halos);
void gpu_pack_edges(GpuState*, double* host_top, double* host_bottom);
void gpu_install_halos(GpuState*, const double* host_top,
                       const double* host_bottom, bool has_up, bool has_down);
void gpu_step_and_swap(GpuState*);
void gpu_synchronize(GpuState*);
void gpu_copy_owned_to_host(GpuState*, double* contiguous_owned_rows);
void gpu_destroy(GpuState*);
```

`gpu_create` allocates and initializes both fields. `gpu_pack_edges` returns only when the host send rows are ready. `gpu_install_halos` guarantees that the subsequent kernel sees the installed halos. `gpu_step_and_swap` queues one update and changes the active handles while preserving stream order. `gpu_copy_owned_to_host` returns only when all owned final rows are ready for MPI gathering. Document buffer lengths and error handling in the concrete header. The CPU and GPU paths should reuse the validated halo-exchange routine where compatible. Concrete MPI function signatures remain pending as described in Section 10.1.

### 6.4 Optional CUDA-aware MPI path

Enable direct device-buffer communication only after verifying support on the target cluster. Complete device production of send rows before MPI reads them; complete receives before kernels consume halos; complete sends before a field can become a writable destination. Record this mode in experiment metadata and compare it separately with the host-staged baseline.

## 7. Correctness and validation

### 7.1 Numerical oracle

Use an independently callable serial CPU reference with the exact initialization, fixed boundaries, stencil, and iteration count in this specification. For MPI and MPI+CUDA, reconstruct the final global grid on rank zero after the timed region and compare against this reference. Broadcast the validation outcome.

For serial validation, also use the one-step check and invariants below; comparing a routine with itself is not sufficient evidence of correctness.

Compute over all N² cells:

```text
max_abs_error = max(abs(result - reference))
relative_l2_error = sqrt(sum((result-reference)^2)) / sqrt(sum(reference^2))
```

If the reference norm is zero, define relative L2 error as zero when the difference norm is zero and infinity otherwise. Accumulate metrics in double precision or higher.

Pass only if every result is finite, all physical boundary values are exactly zero, and for every cell:

```text
abs(result - reference) <= 1e-12 + 1e-10 * abs(reference)
```

Report both aggregate error metrics; relative L2 is diagnostic, not an additional pass threshold. These tolerances are team choices. Do not loosen them solely to hide an unexplained discrepancy. Avoid fast-math options that weaken the stated double-precision contract.

### 7.2 Minimum correctness cases

| Case | Parameters | Purpose |
|---|---|---|
| Initialization only | `N=8, K=0`; `P=1,3` | Hot square, boundaries, gathering, zero swaps |
| Analytic one-step check | `N=8, K=1`; `P=1,3` | With hot square `[2,6)²`, expect `u[3,3]=1.0`, `u[2,3]=0.9`, `u[2,2]=0.8`, `u[1,3]=0.1` |
| Uneven rows and integer bounds | `N=17, K=7`; `P=3,5` | Nondivisible decomposition and hot-square endpoints |
| Tiny local domains | `N=8, K=2`; `P=4,8` | Two-row and one-row ranks |
| Buffer parity | `N=33, K=1,2,9,10`; `P=4` | Correct final buffer for odd/even counts |
| Longer evolution | `N=64, K=100`; `P=1,3,8` | Accumulated error and repeated exchanges |

Run applicable cases for all backends. GPU runs require enough allocated GPUs for the requested ranks; record resource-limited omissions explicitly and run the smaller-rank cases available.

Also check that values stay within `[0,1]` up to rounding tolerance. Invalid-input checks must cover `N<4`, negative steps, malformed arguments, and `P>N`. No correctness claim may rely only on a checksum or a visually plausible heat map.

## 8. Timing protocol

The primary metric is **time for exactly K complete update iterations**.

Exclude argument parsing, allocations, initialization, initial GPU upload, reference computation, final GPU download, global gathering, validation, and file output. Include all per-step halo exchange, host/device row staging, kernels/CPU updates, required waits, and buffer swaps.

- Serial: use a monotonic wall-clock timer around the update loop.
- MPI CPU: complete initialization, call `MPI_Barrier`, record `MPI_Wtime`, run the loop, and record local elapsed time immediately after the loop.
- MPI+CUDA: synchronize the device before the initial barrier; after the loop synchronize the device before recording local elapsed time.
- Reduce local elapsed times with `MPI_MAX` and report the maximum on rank zero. Do not report rank zero's local duration alone or an average over ranks.
- The initial barrier and final timing reduction are outside the timed interval. No final barrier is required before the local end timestamp.
- A `K=0` run is a correctness case, not a meaningful throughput benchmark.

Perform one untimed warm-up run at each benchmark configuration and reset the field to the original initial condition before every measured run. Run at least five measured independent repetitions and report the median and min/max. Keep the iteration count identical within each scaling experiment and large enough for stable timings. Preserve raw repetition records.

Record compiler and flags, source revision, MPI/CUDA versions, CPU/GPU models, nodes, ranks per node, GPU mapping, CPU binding, launch command, N, K, and communication mode. Use one CPU compute thread per rank in the baseline and avoid accidental oversubscription. If reporting end-to-end time, label it separately with explicit endpoints.

## 9. Strong and weak scaling

### 9.1 Strong scaling

Hold global N and K fixed; increase P through feasible allocations, for example `1,2,4,8,...`. Use the same implementation and execution mode within a scaling curve.

```text
speedup(P)    = T(1) / T(P)
efficiency(P) = speedup(P) / P
```

Use one-rank MPI CPU as the baseline for MPI CPU scaling and one-rank/one-GPU MPI+CUDA as the baseline for GPU scaling. Separately report serial-relative acceleration `T_serial/T_parallel`; label it clearly because it mixes hardware and implementation effects. If P=1 does not fit, use a stated `P0` baseline and efficiency `T(P0)*P0/(T(P)*P)`.

### 9.2 Weak scaling for a square global grid

To keep approximate work per rank constant while retaining an N × N grid, choose:

```text
N(P) = nearest integer to N0 * sqrt(P)
K(P) = K0
```

Specify ties as rounding upward. Prefer square process counts such as `P=1,4,9,16` when resources permit, yielding exact integer scaling for integer N0. Check `P<=N(P)`.

For arbitrary P, report actual N, `(N-2)^2/P` interior cells per rank on average, and the minimum/maximum owned interior-cell counts. Rounding and uneven partitions mean the work is approximate. Keep r fixed and K fixed; this experiment measures computational scaling, not convergence to the same physical solution.

```text
weak_efficiency(P) = T(1,N0,K0) / T(P,N(P),K0)
```

**Do not use `N(P)=N0*P` for square grids**: that increases N²/P with P. With this 1D decomposition, row width and halo traffic grow approximately as sqrt(P), while local row count shrinks; discuss this communication cost when interpreting weak scaling.

### 9.3 Required 256-CPU-core MPI experiment

Following the original team experiment plan, the MPI CPU scaling experiments must include a run using **at least 256 CPU cores**. With the baseline of one single-threaded MPI rank per core, this means at least 256 ranks on distinct allocated CPU cores, without oversubscription. Choose `N >= P` to satisfy the decomposition constraint and use the timing and repetition protocol in Section 8.

Run on allocated compute nodes. Record node count, allocated and used CPU cores, MPI rank count, ranks per node, CPU binding, N, K, and raw timings. Include the configuration in the MPI scaling data with its stated baseline. This is a CPU MPI requirement; it does not require 256 GPUs. GPU experiments use the available allocated GPU resources under Section 6.

### 9.4 Experiment deliverables

Keep a manifest of configurations and raw CSV results. Produce runtime, strong speedup/efficiency, and weak runtime/efficiency plots with clearly labeled hardware and baselines. State excluded runs, failures, resource limits, and whether each plotted configuration passed validation. No particular size list or plot count is claimed here as a verified course requirement.

## 10. Integration and pending MPI interfaces

B delivers a compiled, runnable, validated MPI module. C performs MPI+CUDA integration, including connecting GPU staging buffers and implementing GPU timing synchronization. B supports the MPI handoff and fixes defects in the MPI module. The separate team division-of-work PDF defines the full task allocation; this specification does not repeat that allocation.

### 10.1 MPI interface status

Concrete MPI function names, signatures, shared structures, and header declarations are intentionally deferred until B's MPI implementation is complete. Section 4 already defines the required decomposition, data layout, message tags, communication semantics, and gathering behavior; these remain the shared development contract. The CUDA interfaces in Section 6.3 describe semantic behavior and will be aligned with the completed MPI interface during integration.

After the MPI implementation is validated, document its actual interfaces here and in the relevant header, including buffer lengths, ownership and lifetime, communication completion guarantees, communicator usage, and error handling. No new MPI API is prescribed at this stage.

### 10.2 Handoff evidence

The validated MPI handoff includes the source revision, build and run commands, supported environment, interface documentation, small validation results, and known limitations. Include the `N=10,P=3` decomposition example and the `N=17,P=3,K=7` correctness result. MPI+CUDA integration must also demonstrate that both odd and even iteration counts return the active final device buffer.

Shared changes to initialization, r, row ownership, message tags, output schema, tolerances, timing boundaries, or CUDA API require review by the affected contributors before merging. Optimizations must preserve the baseline validation suite and be identifiable in benchmark metadata.

## 11. Definition of done

- [ ] Official handout checked; course requirements and any conflicts recorded.
- [ ] All three executables build with documented commands.
- [ ] Numerical constants, initialization, fixed boundaries, and layouts match this specification.
- [ ] Uneven MPI partitions and one-row local domains work.
- [ ] GPU integration passes the common serial-reference checks on available resources.
- [ ] Requested validation failures and runtime failures return nonzero coherently.
- [ ] Timing includes complete iterations and reports maximum parallel-rank time.
- [ ] Strong/weak scaling records include repetitions, metadata, and explicit baselines.
- [ ] MPI CPU scaling includes at least 256 CPU cores with configuration and timing records.
- [ ] Concrete MPI interfaces are documented after implementation, and the validated module is handed off for MPI+CUDA integration.

This checklist defines team acceptance; it does not substitute for the official course marking rubric.
