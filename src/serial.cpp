#include "../include/heat_common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Configuration for serial execution.
struct Config {
    int n = 0;
    int steps = 0;
    bool validate = false;
    std::string output;
};

// Parse a nonnegative integer safely.
int parse_integer(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Empty integer");
    }

    int result = 0;

    for (char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(
                "Invalid integer: " + text
            );
        }

        int digit = c - '0';

        if (result > (INT_MAX - digit) / 10) {
            throw std::out_of_range("Integer too large");
        }

        result = result * 10 + digit;
    }

    return result;
}

// Print command-line usage.
void print_usage() {
    std::cout
        << "Usage: ./heat_serial --n N --steps K "
        << "[--validate] [--output PATH]\n";
}

// Parse command-line arguments.
Config parse_args(int argc, char* argv[]) {
    Config cfg;

    bool has_n = false;
    bool has_steps = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--n" && i + 1 < argc) {
            if (has_n) {
                throw std::invalid_argument("Duplicate --n");
            }

            cfg.n = parse_integer(argv[++i]);
            has_n = true;
        }
        else if (arg == "--steps" && i + 1 < argc) {
            if (has_steps) {
                throw std::invalid_argument(
                    "Duplicate --steps"
                );
            }

            cfg.steps = parse_integer(argv[++i]);
            has_steps = true;
        }
        else if (arg == "--validate") {
            cfg.validate = true;
        }
        else if (arg == "--output" && i + 1 < argc) {
            cfg.output = argv[++i];

            if (cfg.output.empty()) {
                throw std::invalid_argument(
                    "Empty output path"
                );
            }
        }
        else {
            throw std::invalid_argument(
                "Unknown argument: " + arg
            );
        }
    }

    if (!has_n || !has_steps || cfg.n < 4) {
        throw std::invalid_argument(
            "Require --n N (N >= 4) "
            "and --steps K (K >= 0)"
        );
    }

    return cfg;
}

// Initialize the temperature field using shared utilities.
void initialize(std::vector<double>& grid, int n) {
    heat::initialize_grid(grid, n);
}

// Perform one five-point stencil update.
void serial_step(
    const std::vector<double>& current,
    std::vector<double>& next,
    int n
) {
    for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {

            next[heat::index(i, j, n)] =
                heat::update_cell(current, i, j, n);
        }
    }
}

// Execute exactly the requested number of iterations.
void simulate(
    std::vector<double>& current,
    std::vector<double>& next,
    int n,
    int steps
) {
    for (int t = 0; t < steps; ++t) {
        serial_step(current, next, n);

        // Double buffering.
        current.swap(next);
    }
}

// Independent reference implementation for validation.
std::vector<double> reference_solution(
    int n,
    int steps
) {
    std::size_t cells =
        static_cast<std::size_t>(n) * n;

    std::vector<double> a(cells, 0.0);
    std::vector<double> b(cells, 0.0);

    initialize(a, n);

    for (int t = 0; t < steps; ++t) {
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {

                double center =
                    a[heat::index(i, j, n)];

                double up =
                    a[heat::index(i - 1, j, n)];

                double down =
                    a[heat::index(i + 1, j, n)];

                double left =
                    a[heat::index(i, j - 1, n)];

                double right =
                    a[heat::index(i, j + 1, n)];

                b[heat::index(i, j, n)] =
                    center + heat::R * (
                        up + down + left + right
                        - 4.0 * center
                    );
            }
        }

        a.swap(b);
    }

    return a;
}

// Validation results.
struct ValidationResult {
    bool passed = true;
    double max_abs = 0.0;
    double relative_l2 = 0.0;
};

// Validate numerical correctness.
ValidationResult validate_result(
    const std::vector<double>& result,
    int n,
    int steps
) {
    ValidationResult check;

    std::vector<double> reference =
        reference_solution(n, steps);

    double diff_sq = 0.0;
    double ref_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            std::size_t k = heat::index(i, j, n);

            double actual = result[k];
            double expected = reference[k];

            if (!std::isfinite(actual)) {
                check.passed = false;
                continue;
            }

            double diff = std::abs(actual - expected);

            check.max_abs =
                std::max(check.max_abs, diff);

            diff_sq += diff * diff;
            ref_sq += expected * expected;

            double tolerance =
                1e-12 + 1e-10 * std::abs(expected);

            if (diff > tolerance) {
                check.passed = false;
            }

            // Normalized temperature bounds.
            if (actual < -1e-12 ||
                actual > 1.0 + 1e-12) {
                check.passed = false;
            }

            // Fixed zero boundary conditions.
            bool boundary =
                (i == 0 || i == n - 1 ||
                 j == 0 || j == n - 1);

            if (boundary && actual != 0.0) {
                check.passed = false;
            }
        }
    }

    if (ref_sq > 0.0) {
        check.relative_l2 =
            std::sqrt(diff_sq) / std::sqrt(ref_sq);
    }
    else {
        check.relative_l2 =
            (diff_sq == 0.0)
            ? 0.0
            : std::numeric_limits<double>::infinity();
    }

    // Analytical one-step test from SPEC.md.
    if (n == 8 && steps == 1) {
        const double tol = 1e-12;

        auto matches = [&](int i, int j, double expected) {
            return std::abs(
                result[heat::index(i, j, n)] - expected
            ) <= tol;
        };

        if (!matches(3, 3, 1.0) ||
            !matches(2, 3, 0.9) ||
            !matches(2, 2, 0.8) ||
            !matches(1, 3, 0.1)) {
            check.passed = false;
        }
    }

    // Explicit initialization test.
    if (steps == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (result[heat::index(i, j, n)] !=
                    heat::initial_value(i, j, n)) {
                    check.passed = false;
                }
            }
        }
    }

    return check;
}

// Write the final N x N grid to CSV.
void write_csv(
    const std::string& path,
    const std::vector<double>& grid,
    int n
) {
    std::ofstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open output file: " + path
        );
    }

    file << std::setprecision(17);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            if (j > 0) {
                file << ",";
            }

            file << grid[heat::index(i, j, n)];
        }

        file << "\n";
    }

    if (!file) {
        throw std::runtime_error(
            "Failed to write output CSV"
        );
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 &&
            std::string(argv[1]) == "--help") {
            print_usage();
            return 0;
        }

        Config cfg = parse_args(argc, argv);

        std::size_t width =
            static_cast<std::size_t>(cfg.n);

        // Check allocation size.
        if (width >
            std::numeric_limits<std::size_t>::max()
            / width / sizeof(double)) {

            throw std::overflow_error(
                "Grid allocation overflow"
            );
        }

        std::size_t cells = width * width;

        // Double-buffered temperature grids.
        std::vector<double> current(cells, 0.0);
        std::vector<double> next(cells, 0.0);

        // Initialize before timing.
        initialize(current, cfg.n);

        // Start timing.
        auto start =
            std::chrono::steady_clock::now();

        simulate(current, next, cfg.n, cfg.steps);

        // Stop timing.
        auto finish =
            std::chrono::steady_clock::now();

        double elapsed =
            std::chrono::duration<double>(
                finish - start
            ).count();

        ValidationResult check;
        std::string status = "not_run";

        // Validation is excluded from runtime.
        if (cfg.validate) {
            check = validate_result(
                current, cfg.n, cfg.steps
            );

            status = check.passed ? "pass" : "fail";
        }

        // File output is excluded from runtime.
        if (!cfg.output.empty()) {
            write_csv(cfg.output, current, cfg.n);
        }

        // Machine-readable CSV output.
        std::cout << std::setprecision(17);

        std::cout
            << "implementation,n,steps,r,ranks,gpus,"
            << "elapsed_s,validation,max_abs_error,"
            << "relative_l2_error\n";

        std::cout
            << "serial,"
            << cfg.n << ","
            << cfg.steps << ","
            << heat::R << ","
            << "1,0,"
            << elapsed << ","
            << status << ",";

        if (cfg.validate) {
            std::cout
                << check.max_abs << ","
                << check.relative_l2 << "\n";
        }
        else {
            std::cout << "NA,NA\n";
        }

        if (cfg.validate && !check.passed) {
            std::cerr << "Validation failed\n";
            return 1;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "Error: " << e.what() << "\n";

        print_usage();
        return 1;
    }
}
