// CPU MPI heat diffusion: row decomposition, halo exchange and reference validation.
#include "../include/mpi_interface.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iomanip>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using heat_mpi::Domain;
using heat_mpi::decompose;

int parse_integer(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("Empty integer argument");
    int value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9')
            throw std::invalid_argument("Expected a nonnegative decimal integer: " + text);
        const int digit = ch - '0';
        if (value > (INT_MAX - digit) / 10)
            throw std::out_of_range("Integer exceeds INT_MAX: " + text);
        value = value * 10 + digit;
    }
    return value;
}

std::size_t buffer_size(int n, int rows) {
    const auto width = static_cast<std::size_t>(n);
    const auto height = static_cast<std::size_t>(rows) + 2;
    if (height > std::numeric_limits<std::size_t>::max() / width / sizeof(double))
        throw std::overflow_error("Local grid allocation size overflow");
    return height * width;
}

void initialize(std::vector<double>& current, int n, const Domain& domain) {
    const int lo = n / 4;
    const int hi = static_cast<int>((3LL * n) / 4);
    for (int row = 1; row <= domain.rows; ++row) {
        const int global_row = domain.start + row - 1;
        if (global_row == 0 || global_row == n - 1) continue;
        if (global_row < lo || global_row >= hi) continue;
        for (int col = lo; col < hi; ++col)
            current[static_cast<std::size_t>(row) * n + col] = 1.0;
    }
    // Halos and physical boundaries remain zero. No initial scatter is needed.
}

void update(std::vector<double>& current, std::vector<double>& next,
            int n, const Domain& domain) {
    for (int row = 1; row <= domain.rows; ++row) {
        const int global_row = domain.start + row - 1;
        if (global_row == 0 || global_row == n - 1) continue;
        for (int j = 1; j < n - 1; ++j) {
            const auto k = static_cast<std::size_t>(row) * n + j;
            next[k] = current[k] + 0.1 * (current[k-n] + current[k+n]
                       + current[k-1] + current[k+1] - 4.0 * current[k]);
        }
    }
    current.swap(next);
}

// Independent reference for standalone MPI correctness validation.
// Full global storage and separate loops avoid MPI indexing/decomposition reuse.
std::vector<double> serial_reference(int n, int steps) {
    std::vector<double> field(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> scratch(field.size(), 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i > 0 && i < n-1 && j > 0 && j < n-1 &&
                i >= n/4 && i < 3LL*n/4 && j >= n/4 && j < 3LL*n/4)
                field[static_cast<std::size_t>(i)*n+j] = 1.0;
    for (int step = 0; step < steps; ++step) {
        for (int i = 1; i < n-1; ++i)
            for (int j = 1; j < n-1; ++j) {
                const auto k = static_cast<std::size_t>(i)*n+j;
                scratch[k] = field[k] + 0.1 * (field[k-n] + field[k+n]
                             + field[k-1] + field[k+1] - 4.0*field[k]);
            }
        field.swap(scratch);
    }
    return field;
}

struct Validation {
    int passed = 1;
    double max_abs = 0.0;
    double relative_l2 = 0.0;
};

Validation validate_grid(const std::vector<double>& current,
                           const std::vector<double>& next,
                           int n, int steps, const Domain& domain, int rank, int size) {
    // Small-grid validation uses the classic int-count MPI API.
    const long long cells = 1LL * n * n;
    if (cells > INT_MAX || static_cast<unsigned long long>(cells) >
            std::numeric_limits<std::size_t>::max() / sizeof(double))
        throw std::overflow_error("Validation grid exceeds supported gather/allocation size");

    // Check storage that is deliberately excluded from the global gather.
    int local_storage_ok = 1;
    if (steps == 0)
        for (double value : next) if (value != 0.0) local_storage_ok = 0;
    const auto bottom = (static_cast<std::size_t>(domain.rows) + 1) * n;
    for (int j = 0; j < n; ++j)
        if ((steps == 0 || domain.upper == MPI_PROC_NULL) && current[j] != 0.0)
            local_storage_ok = 0;
    for (int j = 0; j < n; ++j)
        if ((steps == 0 || domain.lower == MPI_PROC_NULL) && current[bottom+j] != 0.0)
            local_storage_ok = 0;
    int storage_ok = 0;
    MPI_Allreduce(&local_storage_ok, &storage_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    std::vector<double> global;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        // NaNs expose any cells accidentally left unwritten by gathering.
        global.assign(static_cast<std::size_t>(cells), std::numeric_limits<double>::quiet_NaN());
        counts.resize(size);
        displacements.resize(size);
        for (int p = 0; p < size; ++p) {
            const Domain part = decompose(n, p, size);
            counts[p] = part.rows * n;
            displacements[p] = part.start * n;
        }
    }
    MPI_Gatherv(current.data() + n, domain.rows * n, MPI_DOUBLE,
                global.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    Validation result;
    result.passed = storage_ok;
    if (rank == 0) {
        const auto reference = serial_reference(n, steps);
        std::size_t mismatches = 0;
        long double difference_squared = 0, reference_squared = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const auto k = static_cast<std::size_t>(i)*n+j;
                const double actual = global[k], expected = reference[k];
                const bool finite = std::isfinite(actual) && std::isfinite(expected);
                const double error = finite ? std::abs(actual-expected)
                                            : std::numeric_limits<double>::infinity();
                result.max_abs = std::max(result.max_abs, error);
                difference_squared += static_cast<long double>(error)*error;
                reference_squared += static_cast<long double>(expected)*expected;
                const bool boundary = i == 0 || j == 0 || i == n-1 || j == n-1;
                if (!finite || error > 1e-12 + 1e-10*std::abs(expected) ||
                    (boundary && actual != 0.0) || actual < -1e-12 || actual > 1.0+1e-12) {
                    if (mismatches < 10)
                        std::cerr << "Mismatch at (" << i << ',' << j << "): expected="
                                  << expected << ", actual=" << actual << '\n';
                    ++mismatches;
                }
            }
        }
        result.relative_l2 = reference_squared == 0
            ? (difference_squared == 0 ? 0.0 : std::numeric_limits<double>::infinity())
            : static_cast<double>(std::sqrt(difference_squared/reference_squared));
        // Analytic values independently anchor both implementations for one step.
        bool analytic_ok = true;
        if (n == 8 && steps == 1) {
            const int indices[] = {3*8+3, 2*8+3, 2*8+2, 1*8+3};
            const double expected[] = {1.0, 0.9, 0.8, 0.1};
            for (int k = 0; k < 4; ++k)
                analytic_ok = analytic_ok && std::abs(global[indices[k]]-expected[k]) <= 1e-12;
            std::cerr << "Analytic one-step check: " << (analytic_ok ? "PASS" : "FAIL") << '\n';
        }
        result.passed = storage_ok && mismatches == 0 && analytic_ok;
        std::cerr << "Validation: " << (result.passed ? "PASS" : "FAIL")
                  << "; checked cells=" << cells << "; mismatches=" << mismatches
                  << "; storage=" << (storage_ok ? "PASS" : "FAIL") << '\n';
    }
    MPI_Bcast(&result.passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int run(int argc, char** argv, int rank, int size) {
    // Root parses once, then tells every rank whether to continue or exit.
    int status = 0; // 0: run, 1: help, 2: invalid/unsupported arguments
    int n = 0;
    int steps = 0;
    int validate = 0;
    if (rank == 0) {
        try {
            if (argc == 2 && std::string(argv[1]) == "--help") {
                status = 1;
                std::cout << "Usage: heat_mpi --n N --steps K [--validate]\n"
                          << "N >= 4, K >= 0 and ranks <= N.\n"
                          << "--output is not implemented yet.\n";
            } else {
                bool have_n = false, have_steps = false;
                for (int i = 1; i < argc; ++i) {
                    const std::string option = argv[i];
                    if (option == "--validate") {
                        if (validate) throw std::invalid_argument("Duplicate --validate");
                        validate = 1;
                        continue;
                    }
                    if (option != "--n" && option != "--steps")
                        throw std::invalid_argument("Unknown or not-yet-supported option: " + option);
                    if (++i == argc) throw std::invalid_argument("Missing value for " + option);
                    const int value = parse_integer(argv[i]);
                    if (option == "--n") {
                        if (have_n) throw std::invalid_argument("Duplicate --n");
                        have_n = true;
                        n = value;
                    } else {
                        if (have_steps) throw std::invalid_argument("Duplicate --steps");
                        have_steps = true;
                        steps = value;
                    }
                }
                if (!have_n || !have_steps)
                    throw std::invalid_argument("Both --n and --steps are required");
                if (n < 4) throw std::invalid_argument("N must be at least 4");
                if (size > n) throw std::invalid_argument("MPI rank count must not exceed N");

            }
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            status = 2;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) return status == 1 ? 0 : 1;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const Domain domain = decompose(n, rank, size);
    const auto count = buffer_size(n, domain.rows);
    std::vector<double> current(count, 0.0);
    std::vector<double> next(count, 0.0);
    initialize(current, n, domain);

    MPI_Barrier(MPI_COMM_WORLD);
    const double started = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        heat_mpi::exchange_halos(current.data(), n, domain);
        update(current, next, n, domain);
    }
    const double local_elapsed = MPI_Wtime() - started;
    double elapsed = 0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    Validation result;
    if (validate) result = validate_grid(current, next, n, steps, domain, rank, size);
    if (rank == 0) {
        std::cout << "implementation,n,steps,r,ranks,gpus,elapsed_s,validation,max_abs_error,relative_l2_error\n"
                  << std::setprecision(17) << "mpi," << n << ',' << steps << ",0.1," << size
                  << ",0," << elapsed << ',';
        if (validate)
            std::cout << (result.passed ? "pass" : "fail") << ',' << result.max_abs
                      << ',' << result.relative_l2 << '\n';
        else std::cout << "not_run,NA,NA\n";
    }
    if (!result.passed) return 1;
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "SPEC requires IEEE 754 64-bit double");
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int result = 1;
    try {
        result = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        // An allocation failure on one rank must not strand others in a collective.
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
