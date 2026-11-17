#include "../include/cuda_host_utils.h"

#include <functional>
#include <iostream>

namespace host = heat_cuda_host;

namespace {
int checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function action, const std::string& message) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

host::Config config(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "heat_mpi_cuda");
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return host::parse_args(static_cast<int>(argv.size()), argv.data());
}

void put(const std::string& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write test fixture");
    file << text;
    file.close();
    if (!file) throw std::runtime_error("Cannot finish test fixture");
}

void test_arguments() {
    const auto valid = config({"--n", "17", "--steps", "7", "--reference", "a.csv",
                               "--block-x", "16", "--block-y", "4"});
    require(valid.n == 17 && valid.steps == 7 && valid.validate &&
            valid.reference == "a.csv" && valid.block_x == 16 && valid.block_y == 4,
            "Valid configuration or reference-implies-validation failed");
    const std::vector<std::vector<std::string>> invalid = {
        {}, {"--n", "8"}, {"--steps", "1"},
        {"--n", "3", "--steps", "1"},
        {"--n", "8", "--steps", "-1"},
        {"--n", "8x", "--steps", "1"},
        {"--n", "2147483648", "--steps", "1"},
        {"--n", "8", "--steps", "1", "--n", "9"},
        {"--n", "8", "--steps", "1", "--validate", "--validate"},
        {"--n", "8", "--steps", "1", "--bad"},
        {"--n", "8", "--steps", "1", "--output"},
        {"--n", "8", "--steps", "1", "--output", "--validate"},
        {"--n", "8", "--steps", "1", "--block-x", "0"},
        {"--n", "8", "--steps", "1", "--block-x", "1024", "--block-y", "2"},
        {"--n", "8", "--steps", "1", "--reference", "a.csv", "--output", "a.csv"},
        {"--n", "8", "--steps", "1", "--profile-output", "a.csv", "--output", "a.csv"},
    };
    for (const auto& arguments : invalid)
        rejects([&] { (void)config(arguments); }, "Invalid CLI was accepted");
    require(host::integer("0") == 0 && host::integer("2147483647") == INT_MAX,
            "Integer endpoints are incorrect");
    rejects([] { (void)host::serial_reference(8, -1); }, "Negative reference steps accepted");
}

void test_analytic_fields() {
    const auto initial = host::serial_reference(8, 0);
    int hot_cells = 0;
    for (double value : initial) if (value == 1.0) ++hot_cells;
    require(initial.size() == 64 && hot_cells == 16, "N8 initial hot square is incorrect");
    const auto once = host::serial_reference(8, 1);
    const auto at = [&](int row, int column) { return once[static_cast<std::size_t>(row) * 8 + column]; };
    require(std::abs(at(3, 3) - 1.0) < 1e-14 && std::abs(at(2, 3) - 0.9) < 1e-14 &&
            std::abs(at(2, 2) - 0.8) < 1e-14 && std::abs(at(1, 3) - 0.1) < 1e-14,
            "N8 one-step analytic values differ");

    // For N=4 the four identical interior cells form an exact mode: 0.8^K.
    // This independent analytic check exercises zero, odd and even buffer swaps.
    for (int steps : {0, 1, 2, 9, 10}) {
        const auto field = host::serial_reference(4, steps);
        const double expected = std::pow(0.8, steps);
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column) {
                const bool inside = row > 0 && row < 3 && column > 0 && column < 3;
                const double value = field[static_cast<std::size_t>(row) * 4 + column];
                require(std::abs(value - (inside ? expected : 0.0)) < 1e-13,
                        "N4 analytic decay/parity differs");
            }
    }

    const auto uneven = host::serial_reference(17, 0);
    int count = 0;
    for (double value : uneven) if (value == 1.0) ++count;
    require(count == 64 && uneven[4 * 17 + 4] == 1.0 && uneven[11 * 17 + 11] == 1.0 &&
            uneven[12 * 17 + 12] == 0.0, "N17 integer hot-square bounds differ");
}

void test_validation() {
    const auto reference = host::serial_reference(8, 1);
    auto bad = reference;
    bad[0] = 5e-13;
    require(heat::compare_results(reference, bad).passed,
            "Fixture should be within the common array tolerance");
    require(!host::validate(bad, reference, 8, 1).passed,
            "Nonzero physical boundary was accepted");
    bad = reference;
    bad[3 * 8 + 3] = std::numeric_limits<double>::quiet_NaN();
    require(!host::validate(bad, reference, 8, 1).passed, "NaN was accepted");
    bad = reference;
    bad[3 * 8 + 3] = 1.0 + 5e-11;
    require(heat::compare_results(reference, bad).passed,
            "Range fixture should remain inside relative comparison tolerance");
    require(!host::validate(bad, reference, 8, 1).passed,
            "Out-of-range temperature was accepted");
    rejects([&] { (void)host::validate(std::vector<double>(3), reference, 8, 1); },
            "Wrong field shape was accepted");
    require(host::validate(reference, reference, 8, 1).passed,
            "Valid analytic field rejected by the physical validator");
}

void test_csv(const std::string& temporary) {
    const auto path = temporary + "/field.csv";
    const auto field = host::serial_reference(17, 7);
    host::write_field(path, field, 17);
    require(host::read_field(path, 17) == field,
            "17-digit field CSV failed exact round trip");
    rejects([&] { host::write_field(path, std::vector<double>(3), 4); },
            "Wrong output shape was accepted");
    const std::string row = "0,0,0,0\n";
    const std::vector<std::string> malformed = {
        row + row + row,                                  // Too few rows.
        row + row + row + row + row,                      // Too many rows.
        "0,0,0\n" + row + row + row,                    // Too few columns.
        "0,0,0,0,0\n" + row + row + row,                // Too many columns.
        "0,0,0,0,\n" + row + row + row,                 // Trailing empty value.
        "0,,0,0\n" + row + row + row,                   // Interior empty value.
        "nan,0,0,0\n" + row + row + row,
        "inf,0,0,0\n" + row + row + row,
        "0junk,0,0,0\n" + row + row + row,
        row + row + row + row + "\n",                   // Extra blank row.
    };
    for (const auto& text : malformed) {
        put(path, text);
        rejects([&] { (void)host::read_field(path, 4); },
                "Malformed reference CSV was accepted");
    }
    put(path, " 0 ,0,0,0\r\n0,0,0,0\r\n0,0,0,0\r\n0,0,0,0\r\n");
    require(host::read_field(path, 4) == std::vector<double>(16, 0.0),
            "Valid whitespace/CRLF CSV was rejected");
}

void test_saved_reference(const std::string& repository, int n, int steps) {
    const auto file = repository + "/results/serial_reference_" +
        std::to_string(n) + "_" + std::to_string(steps) + ".csv";
    const auto saved = host::read_field(file, n);
    const auto independent = host::serial_reference(n, steps);
    const auto metrics = host::validate(independent, saved, n, steps);
    require(metrics.passed, "Independent CPU reference differs from A's saved field");
    std::cout << "reference N=" << n << " K=" << steps
              << " max_abs_error=" << std::setprecision(17) << metrics.max_abs_error << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: cuda_host_tests REPOSITORY_ROOT TEMP_DIRECTORY\n";
        return 2;
    }
    try {
        const std::string repository = argv[1];
        const std::string temporary = argv[2]; // Caller creates this directory.
        test_arguments();
        test_analytic_fields();
        test_validation();
        test_csv(temporary);
        test_saved_reference(repository, 64, 100);
        test_saved_reference(repository, 128, 200);
        std::cout << "PASS: " << checks << " CPU-only checks; no CUDA/MPI runtime was exercised.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
