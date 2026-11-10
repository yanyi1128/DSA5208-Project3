# Vanda CPU experiment workflow

## Confirmed from the account's queue output

- PBS is installed. cpu_parallel has resources_max.ncpus=160, so it cannot
  accept the required single 256-core run. Its default node_pool is cpu_free.
- batch_cpu has resources_max.ncpus=720 and nodect=10, with node_pool=cpu_node.
- Both are from_route_only=True: do not submit directly to either execution queue.
- batch_cpu shows queue_charge_rate=1; this is not a currency price or proof of
  account eligibility. Confirm the course project/allocation and routing policy.
- The listed Open MPI module is openmpi/gcc/64/4.1.8; it was not loaded in the
  supplied module list. Confirm its compiler dependencies before running.

The course comp_res.pdf only provides account/login instructions, not resource
requests. No 256-core allocation has been requested and no job has been submitted.

## Remaining login-node checks (read-only except loading a module)

```bash
qstat -Qf auto
qstat -Qf auto_free
hpc pbs help
module show openmpi/gcc/64/4.1.8
module load openmpi/gcc/64/4.1.8
command -v mpic++ mpirun python3
mpic++ --version
mpirun --version
python3 --version
```

Python 3.8+ is required. Confirm the routing queue, course project identifier,
per-node available CPU count/memory, and CPU binding/site MPI launch guidance.
Do not infer these from the queue's aggregate core limit.

## Scripts already prepared

- scripts/run_mpi.sh launches the standard-library Python benchmark driver.
- scripts/benchmark_mpi.py runs one fresh-process warm-up per configuration,
  followed by five fresh-process measurements. It checks the solver CSV and
  stops on launch/validation errors, preserving earlier results and diagnostics.
- scripts/vanda_mpi.pbs is the PBS job body, not yet a ready-to-submit resource
  request. Queue, select/ncpus/mpiprocs/memory, walltime, and project must be
  supplied via confirmed qsub options. No routing queue is guessed.

The job environment requires MODE, GRID_N, STEPS, RANKS (space separated), and
ALLOCATED_CORES (actual granted/requested total). Optional VALIDATE=1 enables
the serial oracle and is intended for small checks. MPI_MODULE and
MPI_LAUNCH_ARGS can be overridden with the site-approved module/binding setup.
No oversubscription flags are added. MPI mapping must be verified on the first
small allocated run. Run the same resource placement policy across each curve.

## Preview plans without running MPI

From the project root, these commands only print launch plans; the grid sizes
and iteration counts are starting examples, not final benchmark selections:

```bash
# Small end-to-end pilot (submit later within a four-core allocation).
python3 scripts/benchmark_mpi.py --mode strong --n 64 --steps 100 --ranks 1 2 4 --allocated-cores 4 --validate --output results/pilot --dry-run
# Strong scaling example reaching 256 cores.
python3 scripts/benchmark_mpi.py --mode strong --n 4096 --steps 1000 --ranks 1 2 4 8 16 32 64 128 256 --allocated-cores 256 --output results/strong --dry-run
# Weak scaling: N(P) = round(256*sqrt(P)); at P=256, N=4096.
python3 scripts/benchmark_mpi.py --mode weak --n 256 --steps 1000 --ranks 1 2 4 8 16 32 64 128 256 --allocated-cores 256 --output results/weak --dry-run
```

Without --dry-run the driver requires PBS_JOBID. Start with a small pilot,
verify output and binding, then tune K for stable timings and realistic walltime.
Keep K fixed within each curve. Do not run these computations on a login node.

## Results

Each job gets its own directory and binary. data/raw.csv keeps warm-up and all
measured records; data/summary.csv contains medians/minima/maxima and actual
minimum/maximum interior work per rank. Warm-ups are excluded from summaries.
data/metadata.json records commands, revision/dirty state, binary hash, loaded
modules, MPI/compiler version, PBS job details, nodefile and local CPU information.
For mixed nodes, also capture CPU models on each allocated host before claiming
hardware homogeneity. Per-launch stdout/stderr are retained for troubleshooting.

Validation=not_run is preserved for runs without --validate; it is never converted
to pass based on separate small-grid tests. Keep correctness evidence with the
source revision and explicitly distinguish tested configurations in analysis.
Large-grid serial validation may require substantial root memory and time.
Do not commit generated binaries or bulk logs as assignment source deliverables.

## Verification status

Local Python unit tests and dry-run plans can be checked without cluster access.
Actual Bash/PBS execution and MPI launcher integration still require the first
small Vanda allocation. A complete qsub command is pending confirmed routing
and resource settings; the provided PBS body does not choose those settings.


## Confirmed onboarding and submission entry points

The supplied `hpc cpu` example confirms `auto_free` requires no `-P` project,
and project-backed jobs use an assigned project discoverable with `hpc project`.
The supplied module checks confirmed GCC 11.4.1 and Open MPI 4.1.8.
The Python version remains to be checked (`python3 --version`, require >=3.8).
These observations supersede the pending-routing notes above.

After syncing the latest repository onto Vanda, change to the directory containing
src/, include/, scripts/, and SPEC.md. Submit the four-core pilot:

```bash
qsub scripts/vanda_pilot.pbs
qstat -u "$USER"
```

This requests 4 cores, 4 GB memory, and ten minutes in auto_free, then compiles
and runs N=64,K=100 at P=1,2,4 with validation, one warm-up and five measurements
per configuration. Inspect the returned job ID using `qstat -fx JOB_ID` and its
heat_mpi_pilot.oJOB_NUMBER output. Results go under results/runs/JOB_ID_strong/.
The 18 raw CSV records should all have validation=pass; 15 are measured records.
This pilot establishes that the workflow runs, not that small-grid timing is a
useful scaling result.

Only after the pilot succeeds and the actual allocated project is confirmed:

```bash
hpc project
# Replace ACTUAL_PROJECT; these submit jobs, not previews.
bash scripts/submit_vanda_256.sh ACTUAL_PROJECT strong 4096 1000
bash scripts/submit_vanda_256.sh ACTUAL_PROJECT weak 256 1000
```

The 256-core wrapper requests four distinct hosts with 64 cores/ranks and 16 GB
per host, using auto and the supplied project. Vanda's example supports up to
72 cores in a chunk, but actual availability and project entitlement still need
scheduler confirmation. Two-hour walltime and example grid/step sizes should be
adjusted after pilot measurements. The wrapper does not submit either job until
explicitly executed. No remote submission was performed during development.
