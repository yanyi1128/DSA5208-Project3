# MPI module handoff to C

Include `include/mpi_interface.h` and link `src/mpi_interface.cpp`.
Do not link `src/mpi.cpp` into the CUDA program: it contains the CPU executable's
main and independent validation reference. The interface module has no main,
CUDA dependency, initialization/finalization, or allocations.

## Domain and communication

After MPI_Init, call `heat_mpi::make_domain(N, comm)` on each rank.
`rows` is local_rows; `start` is start_row. `upper` and `lower` are neighbors in
that same communicator. The pure `decompose(N, rank, size)` also supports building
gather counts/displacements. It validates inputs and throws invalid_argument;
catch exceptions at the executable boundary and terminate the job coherently.
MPI operation failures in this module abort the communicator. Invalid pointers,
buffer sizes and inconsistent configurations remain caller errors.

CPU layout remains `(rows+2)*N` doubles; owned local rows are 1..rows.
Use `heat_mpi::exchange_halos(current.data(), N, domain, comm)` for this layout.

C uses four host buffers, each N doubles, with receive buffers initially zero:

```cpp
// Complete device-to-host copies into host_top and host_bottom first.
heat_mpi::exchange_rows(host_top, host_bottom, received_top, received_bottom,
                       N, domain, comm);
// All MPI requests are complete here. For real neighbors copy received rows
// into device halos; order these copies before the next stencil kernel.
```

Send buffers may be identical when rows==1. Receive buffers must be disjoint
from one another and from both sends. The caller retains ownership of all memory.
Only host pointers are supported. No CUDA calls or synchronization are performed
by this module. MPI_PROC_NULL receives preserve their buffers. Reserve message
tags 100/101; do not overlap independent exchanges using these tags on comm.

## Compile and recheck on Windows PowerShell

From the repository root:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I "$env:MSMPI_INC" src/mpi.cpp src/mpi_interface.cpp -L "$env:MSMPI_LIB64" -lmsmpi -o heat_mpi.exe
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_mpi.ps1

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I "$env:MSMPI_INC" tests/mpi_interface_test.cpp src/mpi_interface.cpp -L "$env:MSMPI_LIB64" -lmsmpi -o mpi_interface_test.exe
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
mpiexec -n 1 .\mpi_interface_test.exe
mpiexec -n 3 .\mpi_interface_test.exe
mpiexec -n 4 .\mpi_interface_test.exe
```

On Linux use `mpic++ -std=c++17 -O2 src/mpi.cpp src/mpi_interface.cpp -o heat_mpi`.
For CUDA, compile mpi_interface.cpp with the MPI C++ wrapper into an object and
link it with C's CUDA objects and the cluster's MPI libraries; exact CUDA link
flags depend on the installed MPI toolchain.

The standalone staged-buffer test checks repeated exchanges, absent neighbors,
one-row domains, uneven partitions and a communicator with reversed rank order.
The CPU suite checks end-to-end numerical behavior through the contiguous-field
adapter. Previous monolithic CPU tests passed on the user's terminal. This
refactoring and the new interface test still require compilation and execution;
no new passing runtime result is claimed.
