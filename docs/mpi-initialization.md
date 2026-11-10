# CPU MPI correctness milestone

The solver now supports `--n N --steps K [--validate]` for K >= 0.
Each iteration exchanges full halo rows using receive-first nonblocking MPI,
waits for all four requests, updates globally interior points, and swaps buffers.
Physical boundaries remain zero. One-row domains and uneven partitions are supported.
`--output` remains unimplemented and is rejected explicitly.

## Build on Windows PowerShell

From the repository directory, using the installed MinGW and Microsoft MPI SDK:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I "$env:MSMPI_INC" src/mpi.cpp src/mpi_interface.cpp -L "$env:MSMPI_LIB64" -lmsmpi -o heat_mpi.exe
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed; do not run an old executable' }
mpiexec -n 3 .\heat_mpi.exe --n 8 --steps 1 --validate
.\scripts\test_mpi.ps1
```

On Linux with MPI installed:

```sh
mpic++ -std=c++17 -O2 -Wall -Wextra -Wpedantic src/mpi.cpp src/mpi_interface.cpp -o heat_mpi
mpirun -np 3 ./heat_mpi --n 8 --steps 1 --validate
```

## Validation

Rank zero gathers owned rows only and compares every cell against a separate
full-grid serial reference. This temporary reference lives inside mpi.cpp and
supports independent validation without waiting for A; it does not edit
A's serial.cpp. The tolerance is `1e-12 + 1e-10 * abs(reference)`.
Validation also checks finite values, exact zero physical boundaries, the [0,1]
range (with rounding tolerance), and exterior halos. At K=0, all halos and next
must be zero. Interior halos need not be zero after communication.

For N=8,K=1, four analytically known values (1.0, 0.9, 0.8, 0.1) are checked
independently. The PowerShell suite covers 15 valid cases from SPEC, including
odd/even steps, P=N, uneven partitions and 100 iterations, plus invalid inputs.
Validation requires the global grid to fit root memory and N*N <= INT_MAX.
Use small grids for correctness testing; reference computation is serial.

CSV on stdout follows SPEC and contains elapsed time, validation status, maximum
absolute error and relative L2 error. Human-readable checks go to stderr.
Timing includes all update-loop communication and computation and reports the
maximum rank time; gathering/reference checks are excluded. These individual
runs are not the warm-up/repetition protocol for formal benchmarks.

## Verification status

The user previously compiled and passed initialization checks on Windows for
(N,P)=(10,3),(8,1),(17,3),(4,4). The new iteration implementation has not yet been
compiled or run in this session: the agent's g++ still fails at startup with a
DLL relocation error. Run the commands above against newly compiled source.

## Next steps

After the correctness suite passes, integrate A's reference, document the public
MPI interfaces for C, and implement optional field output and benchmark scripts.
