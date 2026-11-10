// Tests C's separate host staging buffers without requiring CUDA.
#include "../include/mpi_interface.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int ok = 1;
    // Reverse rank order to catch accidental use of MPI_COMM_WORLD neighbors.
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, size - rank, &comm);
    MPI_Comm_rank(comm, &rank);
    for (int n : {size < 4 ? 4 : size, size * 2 + 5}) {
        auto domain = heat_mpi::make_domain(n, comm);
        std::vector<double> top(n), bottom(n), up(n, 0), down(n, 0);
        for (int iteration = 0; iteration < 3; ++iteration) {
            // Values encode global row/column and iteration, independently of rank.
            auto value = [=](int row, int col) { return iteration*100000.0 + row*n + col + 1; };
            for (int j = 0; j < n; ++j) {
                top[j] = value(domain.start, j);
                bottom[j] = value(domain.start + domain.rows - 1, j);
            }
            heat_mpi::exchange_rows(top.data(), domain.rows == 1 ? top.data() : bottom.data(),
                                    up.data(), down.data(), n, domain, comm);
            for (int j = 0; j < n; ++j) {
                const double expected_up = domain.upper == MPI_PROC_NULL ? 0 : value(domain.start-1, j);
                const double expected_down = domain.lower == MPI_PROC_NULL ? 0 : value(domain.start+domain.rows, j);
                if (up[j] != expected_up || down[j] != expected_down) ok = 0;
            }
        }
    }
    int all_ok;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (rank == 0) std::cout << "Staged halo interface: " << (all_ok ? "PASS" : "FAIL") << '\n';
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
