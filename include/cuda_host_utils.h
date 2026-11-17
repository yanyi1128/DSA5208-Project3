#ifndef HEAT_CUDA_HOST_UTILS_H
#define HEAT_CUDA_HOST_UTILS_H

// CPU-only helpers for C's executable. A/B's public headers remain unchanged.
#include "heat_common.h"
#include "validation.h"

#include <climits>
#include <fstream>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <string>

namespace heat_cuda_host {

struct Config {
    int n = 0;
    int steps = 0;
    int block_x = 32;
    int block_y = 8;
    bool validate = false;
    std::string output;
    std::string reference;
    std::string profile_output;
};

inline int integer(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("Empty integer argument");
    int value = 0;
    for (char c : text) {
        if (c < '0' || c > '9')
            throw std::invalid_argument("Expected a nonnegative integer: " + text);
        if (value > (INT_MAX - (c - '0')) / 10)
            throw std::out_of_range("Integer exceeds INT_MAX");
        value = value * 10 + (c - '0');
    }
    return value;
}

inline std::size_t grid_cells(int n) {
    if (n < 4) throw std::invalid_argument("N must be at least 4");
    const auto width = static_cast<std::size_t>(n);
    if (width > std::numeric_limits<std::size_t>::max() / width / sizeof(double))
        throw std::overflow_error("Grid allocation size overflow");
    return width * width;
}

inline Config parse_args(int argc, char** argv) {
    Config cfg;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (!seen.insert(option).second)
            throw std::invalid_argument("Duplicate argument: " + option);
        if (option == "--validate") {
            cfg.validate = true;
            continue;
        }
        if (option != "--n" && option != "--steps" && option != "--output" &&
            option != "--reference" && option != "--profile-output" &&
            option != "--block-x" && option != "--block-y")
            throw std::invalid_argument("Unknown argument: " + option);
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + option);
        const std::string value = argv[++i];
        if (value.empty() || value.rfind("--", 0) == 0)
            throw std::invalid_argument("Missing value for " + option);
        if (option == "--n") cfg.n = integer(value);
        else if (option == "--steps") cfg.steps = integer(value);
        else if (option == "--block-x") cfg.block_x = integer(value);
        else if (option == "--block-y") cfg.block_y = integer(value);
        else if (option == "--output") cfg.output = value;
        else if (option == "--reference") cfg.reference = value;
        else cfg.profile_output = value;
    }
    if (!seen.count("--n") || !seen.count("--steps"))
        throw std::invalid_argument("Both --n N and --steps K are required");
    (void)grid_cells(cfg.n);
    if (cfg.block_x < 1 || cfg.block_y < 1 ||
        static_cast<long long>(cfg.block_x) * cfg.block_y > 1024)
        throw std::invalid_argument("Block dimensions must be positive and contain at most 1024 threads");
    if (!cfg.reference.empty()) cfg.validate = true;
    if ((!cfg.output.empty() && cfg.output == cfg.reference) ||
        (!cfg.profile_output.empty() && (cfg.profile_output == cfg.reference ||
                                        cfg.profile_output == cfg.output)))
        throw std::invalid_argument("Reference, field output and profile output must use different paths");
    return cfg;
}

// Independently expanded CPU stencil for optional small-grid validation.
// --reference supports direct comparison with A's heat_serial --output results.
inline std::vector<double> serial_reference(int n, int steps) {
    if (steps < 0) throw std::invalid_argument("Steps must be nonnegative");
    std::vector<double> field(grid_cells(n), 0.0), next(field.size(), 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            field[static_cast<std::size_t>(i) * n + j] = heat::initial_value(i, j, n);
    for (int t = 0; t < steps; ++t) {
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                const auto k = static_cast<std::size_t>(i) * n + j;
                const double center = field[k];
                next[k] = center + heat::R * (field[k - n] + field[k + n] +
                                              field[k - 1] + field[k + 1] - 4.0 * center);
            }
        }
        field.swap(next);
    }
    return field;
}

inline std::vector<double> read_field(const std::string& path, int n) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open serial reference: " + path);
    std::vector<double> field(grid_cells(n));
    std::string row, token;
    for (int i = 0; i < n; ++i) {
        if (!std::getline(input, row) || row.empty())
            throw std::runtime_error("Reference CSV must contain exactly N nonempty rows");
        if (!row.empty() && row.back() == '\r') row.pop_back();
        if (row.empty() || row.back() == ',')
            throw std::runtime_error("Reference CSV contains an empty value");
        std::istringstream columns(row);
        for (int j = 0; j < n; ++j) {
            if (!std::getline(columns, token, ','))
                throw std::runtime_error("Reference CSV has too few columns");
            std::istringstream number(token);
            number.imbue(std::locale::classic());
            double value = 0.0;
            if (!(number >> value) || !std::isfinite(value))
                throw std::runtime_error("Reference CSV contains an invalid or nonfinite number");
            number >> std::ws;
            if (!number.eof()) throw std::runtime_error("Invalid trailing characters in reference CSV");
            field[static_cast<std::size_t>(i) * n + j] = value;
        }
        if (std::getline(columns, token, ','))
            throw std::runtime_error("Reference CSV has too many columns");
    }
    if (std::getline(input, row)) throw std::runtime_error("Reference CSV has too many rows");
    if (!input.eof()) throw std::runtime_error("Failed to read reference CSV");
    return field;
}

inline void write_field(const std::string& path, const std::vector<double>& field, int n) {
    if (field.size() != grid_cells(n)) throw std::invalid_argument("Output grid size mismatch");
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot open field output: " + path);
    output.imbue(std::locale::classic());
    output << std::setprecision(17);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j) output << ',';
            output << field[static_cast<std::size_t>(i) * n + j];
        }
        output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("Failed to write field output: " + path);
}

inline heat::ErrorMetrics validate(const std::vector<double>& field,
                                   const std::vector<double>& reference,
                                   int n, int steps) {
    if (field.size() != grid_cells(n) || reference.size() != field.size())
        throw std::invalid_argument("Validation grid size mismatch");
    auto metrics = heat::compare_results(reference, field);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double value = field[static_cast<std::size_t>(i) * n + j];
            if (!std::isfinite(value) || value < -1e-12 || value > 1.0 + 1e-12)
                metrics.passed = false;
            if ((i == 0 || j == 0 || i == n - 1 || j == n - 1) && value != 0.0)
                metrics.passed = false;
            if (steps == 0 && value != heat::initial_value(i, j, n))
                metrics.passed = false;
        }
    }
    if (n == 8 && steps == 1) {
        const auto check = [&](int i, int j, double expected) {
            return std::abs(field[static_cast<std::size_t>(i) * n + j] - expected) <= 1e-12;
        };
        if (!check(3, 3, 1.0) || !check(2, 3, 0.9) ||
            !check(2, 2, 0.8) || !check(1, 3, 0.1)) metrics.passed = false;
    }
    return metrics;
}

} // namespace heat_cuda_host
#endif
