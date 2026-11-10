#ifndef HEAT_MPI_INTERFACE_H
#define HEAT_MPI_INTERFACE_H

#include <mpi.h>

namespace heat_mpi {

// All ranks/neighbor IDs are relative to the communicator used for exchange.
struct Domain {
    int rows;  // Owned rows, excluding halos.
    int start; // First global row, zero-based.
    int upper;
    int lower;
};

constexpr int TAG_TO_UP = 100;
constexpr int TAG_TO_DOWN = 101;

// Pure calculation. Requires n >= 4, 1 <= size <= n, 0 <= rank < size.
// Throws std::invalid_argument on invalid input; caller coordinates job failure.
Domain decompose(int n, int rank, int size);

// Queries rank/size from an initialized MPI intracommunicator; does not own it.
Domain make_domain(int n, MPI_Comm comm = MPI_COMM_WORLD);

// Each buffer addresses n host-accessible doubles. Receives must not overlap
// each other or either send buffer. Send buffers may coincide (one owned row).
// Initialize receive buffers to zero: MPI_PROC_NULL leaves them unchanged.
// All participating ranks call in the same iteration order with the same n.
// No allocation, CUDA operations, barrier, or buffer ownership transfer occurs.
// Returns only after ALL sends/receives complete. Reserve tags 100/101 on comm;
// do not interleave other exchanges with these tags on the same communicator.
// Device-to-host copies must finish before entry; received host rows may be
// copied to the device after return. Direct device pointers are unsupported.
void exchange_rows(const double* send_top, const double* send_bottom,
                   double* receive_top, double* receive_bottom,
                   int n, const Domain& domain, MPI_Comm comm = MPI_COMM_WORLD);

// Convenience adapter for a contiguous CPU field of (domain.rows+2)*n doubles.
// Row 0 and row rows+1 are halos; rows 1..rows are owned. Same guarantees above.
void exchange_halos(double* current, int n, const Domain& domain,
                    MPI_Comm comm = MPI_COMM_WORLD);

} // namespace heat_mpi
#endif
