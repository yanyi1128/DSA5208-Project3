
#include "include/validation.h"
#include <iostream>
#include <vector>

int main() {
    std::vector<double> reference = {
        0.0, 0.2, 0.5, 1.0
    };

    // Test 1: Identical results
    std::vector<double> result1 = {
        0.0, 0.2, 0.5, 1.0
    };

    auto test1 = heat::compare_results(
        reference, result1
    );

    std::cout << "Test 1 (identical): "
              << (test1.passed ? "PASS" : "FAIL")
              << "\n";

    // Test 2: Small numerical differences
    std::vector<double> result2 = {
        0.0, 0.2, 0.5 + 1e-13, 1.0
    };

    auto test2 = heat::compare_results(
        reference, result2
    );

    std::cout << "Test 2 (small error): "
              << (test2.passed ? "PASS" : "FAIL")
              << "\n";

    // Test 3: Significant numerical differences
    std::vector<double> result3 = {
        0.0, 0.2, 0.8, 1.0
    };

    auto test3 = heat::compare_results(
        reference, result3
    );

    std::cout << "Test 3 (large error detected): "
              << (!test3.passed ? "PASS" : "FAIL")
              << "\n";

    return (test1.passed &&
            test2.passed &&
            !test3.passed) ? 0 : 1;
}
