#ifndef VALIDATION_H
#define VALIDATION_H
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace heat {

struct ErrorMetrics {
    double max_abs_error = 0.0;
    double relative_l2_error = 0.0;
    bool passed = false;
};

// Compare a parallel result against the serial reference.
inline ErrorMetrics compare_results(
    const std::vector<double>& reference,
    const std::vector<double>& result,
    double atol = 1e-12,
    double rtol = 1e-10
) {
    if (reference.size() != result.size()) {
        throw std::invalid_argument(
            "Reference and result sizes do not match"
        );
    }

    ErrorMetrics metrics;
    metrics.passed = true;

    double diff_squared = 0.0;
    double reference_squared = 0.0;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        double expected = reference[i];
        double actual = result[i];

        if (!std::isfinite(expected) ||
            !std::isfinite(actual)) {
            metrics.passed = false;
            metrics.max_abs_error =
                std::numeric_limits<double>::infinity();
            metrics.relative_l2_error =
                std::numeric_limits<double>::infinity();
            return metrics;
        }

        double error = std::abs(actual - expected);

        metrics.max_abs_error =
            std::max(metrics.max_abs_error, error);

        diff_squared += error * error;
        reference_squared += expected * expected;

        double tolerance =
            atol + rtol * std::abs(expected);

        if (error > tolerance) {
            metrics.passed = false;
        }
    }

    if (reference_squared > 0.0) {
        metrics.relative_l2_error =
            std::sqrt(diff_squared / reference_squared);
    }
    else {
        metrics.relative_l2_error =
            (diff_squared == 0.0)
            ? 0.0
            : std::numeric_limits<double>::infinity();
    }

    return metrics;
}

} // namespace heat

#endif
