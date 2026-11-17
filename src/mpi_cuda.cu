#include "cuda_solver.h"
#include "cuda_host_utils.h"
#include "mpi_interface.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstring>
#include <iostream>
#include <memory>

namespace {
void mpi_check(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(error, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void usage() {
    std::cout << "Usage: heat_mpi_cuda --n N --steps K [--validate] [--output PATH]\n"
              << "       [--reference SERIAL_CSV] [--block-x X --block-y Y]\n"
              << "       [--profile-output PATH]\n"
              << "One MPI rank per GPU. --reference also enables validation.\n"
              << "--profile-output enables synchronized diagnostic timing; benchmark separately.\n";
}

void broadcast_string(std::string& value, int rank) {
    if (value.size() > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("Path is too long for MPI broadcast");
    int size = static_cast<int>(value.size());
    mpi_check(MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD), "Broadcast string length");
    if (rank != 0) value.resize(static_cast<std::size_t>(size));
    mpi_check(MPI_Bcast(value.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD), "Broadcast string");
}

// Honour per-rank CUDA_VISIBLE_DEVICES masks, or map node-local rank to the
// shared visible list. UUID checks reject accidental sharing of a local GPU.
int select_device(int rank) {
    MPI_Comm local_comm = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &local_comm), "Create node-local communicator");
    int local_rank = 0, local_size = 0, visible = 0;
    mpi_check(MPI_Comm_rank(local_comm, &local_rank), "Get local rank");
    mpi_check(MPI_Comm_size(local_comm, &local_size), "Get local size");
    cuda_check(cudaGetDeviceCount(&visible), "Get visible CUDA devices");
    if (visible < 1) throw std::runtime_error("No visible GPU; run in an allocated GPU job");
    const int device = visible == 1 ? 0 : local_rank;
    if (device >= visible)
        throw std::runtime_error("More local MPI ranks than visible GPUs");
    cuda_check(cudaSetDevice(device), "Select CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "Get CUDA device properties");
    const int uuid_bytes = static_cast<int>(sizeof(properties.uuid.bytes));
    std::vector<char> all_uuids(static_cast<std::size_t>(local_size) * uuid_bytes);
    mpi_check(MPI_Allgather(properties.uuid.bytes, uuid_bytes, MPI_BYTE, all_uuids.data(),
                            uuid_bytes, MPI_BYTE, local_comm), "Verify exclusive GPU mapping");
    for (int a = 0; a < local_size; ++a)
        for (int b = a + 1; b < local_size; ++b)
            if (std::memcmp(all_uuids.data() + static_cast<std::size_t>(a) * uuid_bytes,
                            all_uuids.data() + static_cast<std::size_t>(b) * uuid_bytes,
                            static_cast<std::size_t>(uuid_bytes)) == 0)
                throw std::runtime_error("Two local MPI ranks selected the same GPU; fix rank/device assignment");
    char hostname[MPI_MAX_PROCESSOR_NAME] = {};
    int hostname_length = 0;
    mpi_check(MPI_Get_processor_name(hostname, &hostname_length), "Get node name");
    std::ostringstream description;
    description << "CUDA_DEVICE rank=" << rank << " local_rank=" << local_rank
                << " host=" << std::string(hostname, hostname_length)
                << " device=" << device << " name=" << properties.name << " uuid=";
    description << std::hex << std::setfill('0');
    for (char byte : properties.uuid.bytes)
        description << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(byte));
    description << '\n';
    std::cerr << description.str();
    mpi_check(MPI_Comm_free(&local_comm), "Free node-local communicator");
    return device;
}

std::vector<double> gather_field(const std::vector<double>& owned, int n,
                                 int rank, int size) {
    const auto cells = heat_cuda_host::grid_cells(n);
    if (cells > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("Validation/output with MPI_Gatherv requires N*N <= INT_MAX");
    std::vector<int> counts, offsets;
    std::vector<double> full;
    if (rank == 0) {
        full.resize(cells);
        counts.resize(size);
        offsets.resize(size);
        for (int p = 0; p < size; ++p) {
            const auto domain = heat_mpi::decompose(n, p, size);
            counts[p] = domain.rows * n;
            offsets[p] = domain.start * n;
        }
    }
    mpi_check(MPI_Gatherv(owned.data(), static_cast<int>(owned.size()), MPI_DOUBLE,
                          full.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD), "Gather final field");
    return full;
}

void write_profiles(const std::string& path, const heat_cuda_host::Config& cfg,
                    heat_cuda::State* gpu, int device, int rank, int size,
                    double mpi_s, double total_s) {
    const auto measured = heat_cuda::profile(gpu);
    const double values[5] = {measured.kernel_s, measured.d2h_s, measured.h2d_s, mpi_s, total_s};
    std::vector<double> all_values;
    std::vector<int> all_devices;
    if (rank == 0) { all_values.resize(static_cast<std::size_t>(size) * 5); all_devices.resize(size); }
    mpi_check(MPI_Gather(values, 5, MPI_DOUBLE, all_values.data(), 5, MPI_DOUBLE,
                         0, MPI_COMM_WORLD), "Gather profile times");
    mpi_check(MPI_Gather(&device, 1, MPI_INT, all_devices.data(), 1, MPI_INT,
                         0, MPI_COMM_WORLD), "Gather profile devices");
    if (rank == 0) {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Cannot open profile output: " + path);
        output.imbue(std::locale::classic());
        output << "rank,device,n,steps,block_x,block_y,kernel_s,d2h_s,h2d_s,mpi_s,total_s\n"
               << std::setprecision(17);
        for (int p = 0; p < size; ++p) {
            output << p << ',' << all_devices[p] << ',' << cfg.n << ',' << cfg.steps
                   << ',' << cfg.block_x << ',' << cfg.block_y;
            for (int i = 0; i < 5; ++i) output << ',' << all_values[static_cast<std::size_t>(p) * 5 + i];
            output << '\n';
        }
        output.close();
        if (!output) throw std::runtime_error("Failed to write profile output");
    }
}

int run(int argc, char** argv, int rank, int size) {
    heat_cuda_host::Config cfg;
    int parse_status = 0; // 0=run, 1=help, 2=invalid
    if (rank == 0) {
        try {
            if (argc == 2 && std::string(argv[1]) == "--help") { usage(); parse_status = 1; }
            else cfg = heat_cuda_host::parse_args(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            usage();
            parse_status = 2;
        }
    }
    mpi_check(MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD), "Broadcast parse status");
    if (parse_status) return parse_status == 1 ? 0 : 2;
    int fields[5] = {cfg.n, cfg.steps, cfg.block_x, cfg.block_y, cfg.validate ? 1 : 0};
    mpi_check(MPI_Bcast(fields, 5, MPI_INT, 0, MPI_COMM_WORLD), "Broadcast configuration");
    cfg.n = fields[0]; cfg.steps = fields[1]; cfg.block_x = fields[2];
    cfg.block_y = fields[3]; cfg.validate = fields[4] != 0;
    broadcast_string(cfg.output, rank);
    broadcast_string(cfg.reference, rank);
    broadcast_string(cfg.profile_output, rank);
    const auto domain = heat_mpi::make_domain(cfg.n);
    const bool collect = cfg.validate || !cfg.output.empty();
    if (collect && heat_cuda_host::grid_cells(cfg.n) > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("Validation/output requires N*N <= INT_MAX; benchmark without gathering");

    // Parse the supplied serial field before reserving GPU buffers or timing.
    std::vector<double> reference;
    if (rank == 0 && !cfg.reference.empty()) reference = heat_cuda_host::read_field(cfg.reference, cfg.n);
    const int device = select_device(rank);
    using Gpu = std::unique_ptr<heat_cuda::State, decltype(&heat_cuda::destroy)>;
    Gpu gpu(heat_cuda::create({cfg.n, domain.rows, domain.start}, device,
                              cfg.block_x, cfg.block_y, !cfg.profile_output.empty()),
            &heat_cuda::destroy);
    const auto edges = heat_cuda::edge_buffers(gpu.get());
    heat_cuda::synchronize(gpu.get());
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Initial timing barrier");
    double mpi_s = 0.0;
    const double start = MPI_Wtime();
    for (int step = 0; step < cfg.steps; ++step) {
        if (size > 1) {
            heat_cuda::pack_edges(gpu.get());
            const double mpi_start = cfg.profile_output.empty() ? 0.0 : MPI_Wtime();
            heat_mpi::exchange_rows(edges.send_top, edges.send_bottom,
                                    edges.receive_top, edges.receive_bottom, cfg.n, domain);
            if (!cfg.profile_output.empty()) mpi_s += MPI_Wtime() - mpi_start;
            heat_cuda::install_halos(gpu.get(), domain.upper != MPI_PROC_NULL,
                                     domain.lower != MPI_PROC_NULL);
        }
        heat_cuda::step_and_swap(gpu.get());
    }
    heat_cuda::synchronize(gpu.get());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    mpi_check(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD), "Reduce iteration time");

    if (!cfg.profile_output.empty()) {
        write_profiles(cfg.profile_output, cfg, gpu.get(), device, rank, size, mpi_s, local_elapsed);
        if (rank == 0) std::cerr << "PROFILE: synchronized diagnostic run; do not mix with normal benchmark times.\n";
    }
    std::vector<double> full;
    if (collect) full = gather_field(heat_cuda::copy_owned_to_host(gpu.get()), cfg.n, rank, size);
    heat::ErrorMetrics metrics;
    int validation_passed = 1;
    if (rank == 0) {
        if (cfg.validate) {
            if (cfg.reference.empty()) reference = heat_cuda_host::serial_reference(cfg.n, cfg.steps);
            metrics = heat_cuda_host::validate(full, reference, cfg.n, cfg.steps);
            validation_passed = metrics.passed ? 1 : 0;
            std::cerr << "VALIDATION reference=" << (cfg.reference.empty() ? "independent_cpu" : cfg.reference)
                      << " status=" << (metrics.passed ? "pass" : "fail") << '\n';
        }
        if (!cfg.output.empty()) heat_cuda_host::write_field(cfg.output, full, cfg.n);
        std::cout << "implementation,n,steps,r,ranks,gpus,elapsed_s,validation,max_abs_error,relative_l2_error\n"
                  << std::setprecision(17) << "mpi_cuda," << cfg.n << ',' << cfg.steps
                  << ',' << heat::R << ',' << size << ',' << size << ',' << elapsed << ',';
        if (cfg.validate)
            std::cout << (metrics.passed ? "pass" : "fail") << ',' << metrics.max_abs_error
                      << ',' << metrics.relative_l2_error << '\n';
        else std::cout << "not_run,NA,NA\n";
    }
    mpi_check(MPI_Bcast(&validation_passed, 1, MPI_INT, 0, MPI_COMM_WORLD), "Broadcast validation result");
    return validation_passed ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed\n";
        return 2;
    }
    int rank = 0, size = 0;
    try {
        mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "Set MPI error handler");
        mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Get rank");
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &size), "Get size");
        const int status = run(argc, argv, rank, size);
        mpi_check(MPI_Finalize(), "MPI_Finalize");
        return status;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
