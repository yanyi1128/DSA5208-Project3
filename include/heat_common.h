#ifndef HEAT_COMMON_H
#define HEAT_COMMON_H

#include <algorithm>
#include <cstddef>
#include <vector>

namespace heat {

constexpr double R = 0.1;

// Row-major indexing.
inline std::size_t index(int i, int j, int n) {
    return static_cast<std::size_t>(i) * n + j;
}

// Same initial condition for Serial, MPI and CUDA.
inline double initial_value(int i, int j, int n) {
    const int lo = n / 4;
    const int hi = static_cast<int>((3LL * n) / 4);

    if (i == 0 || i == n - 1 ||
        j == 0 || j == n - 1) {
        return 0.0;
    }

    return (i >= lo && i < hi &&
            j >= lo && j < hi) ? 1.0 : 0.0;
}

// Initialize the full serial grid.
inline void initialize_grid(
    std::vector<double>& grid, int n
) {
    std::fill(grid.begin(), grid.end(), 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            grid[index(i, j, n)] =
                initial_value(i, j, n);
        }
    }
}

// Five-point stencil applied to one interior cell.
inline double update_cell(
    const std::vector<double>& current,
    int i, int j, int n
) {
    std::size_t k = index(i, j, n);

    return current[k] + R * (
        current[k - n] +
        current[k + n] +
        current[k - 1] +
        current[k + 1] -
        4.0 * current[k]
    );
}

} // namespace heat

#endif
