#include "../include/mpi_interface.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace heat_mpi {
namespace {
void check(int error, MPI_Comm comm) {
    if (error != MPI_SUCCESS) {
        MPI_Abort(comm, error);
        std::abort(); // Guard against an MPI implementation returning from Abort.
    }
}
} // namespace

Domain decompose(int n, int rank, int size) {
    if (n < 4 || size < 1 || size > n || rank < 0 || rank >= size)
        throw std::invalid_argument("Invalid MPI row decomposition parameters");
    const int base = n / size, remainder = n % size;
    return {base + (rank < remainder ? 1 : 0),
            rank * base + std::min(rank, remainder),
            rank == 0 ? MPI_PROC_NULL : rank - 1,
            rank == size - 1 ? MPI_PROC_NULL : rank + 1};
}

Domain make_domain(int n, MPI_Comm comm) {
    int rank = 0, size = 0;
    check(MPI_Comm_rank(comm, &rank), comm);
    check(MPI_Comm_size(comm, &size), comm);
    return decompose(n, rank, size);
}

void exchange_rows(const double* send_top, const double* send_bottom,
                   double* receive_top, double* receive_bottom,
                   int n, const Domain& domain, MPI_Comm comm) {
    MPI_Request requests[4];
    check(MPI_Irecv(receive_top, n, MPI_DOUBLE, domain.upper, TAG_TO_DOWN,
                    comm, &requests[0]), comm);
    check(MPI_Irecv(receive_bottom, n, MPI_DOUBLE, domain.lower, TAG_TO_UP,
                    comm, &requests[1]), comm);
    check(MPI_Isend(send_top, n, MPI_DOUBLE, domain.upper, TAG_TO_UP,
                    comm, &requests[2]), comm);
    check(MPI_Isend(send_bottom, n, MPI_DOUBLE, domain.lower, TAG_TO_DOWN,
                    comm, &requests[3]), comm);
    check(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE), comm);
}

void exchange_halos(double* current, int n, const Domain& domain, MPI_Comm comm) {
    const auto last = static_cast<std::size_t>(domain.rows) * n;
    exchange_rows(current + n, current + last, current, current + last + n,
                  n, domain, comm);
}
} // namespace heat_mpi
