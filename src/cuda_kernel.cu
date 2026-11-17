#include "../include/cuda_solver.h"
#include "../include/heat_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace heat_cuda {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* description) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(description);
    }
    return a * b;
}

// Both dimensions are grid-stride loops so even a device's small maxGridSize[1]
// cannot truncate a tall local domain. Each (row, column) has exactly one writer.
__global__ void heat_stencil(const double* current, double* next,
                             int n, int rows, int start, double r) {
    const std::size_t width = static_cast<std::size_t>(n);
    const std::size_t first_column =
        1 + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t first_row =
        1 + static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t column_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t row_stride =
        static_cast<std::size_t>(gridDim.y) * blockDim.y;

    for (std::size_t row = first_row;
         row <= static_cast<std::size_t>(rows); row += row_stride) {
        const std::size_t global_row =
            static_cast<std::size_t>(start) + row - 1;
        if (global_row == 0 || global_row == width - 1) {
            continue;
        }
        for (std::size_t column = first_column;
             column < width - 1; column += column_stride) {
            const std::size_t k = row * width + column;
            next[k] = current[k] + r *
                (current[k - width] + current[k + width] +
                 current[k - 1] + current[k + 1] - 4.0 * current[k]);
        }
    }
}

}  // namespace

struct State {
    Domain domain{};
    int device = -1;
    std::size_t row_bytes = 0;
    std::size_t owned_cells = 0;
    dim3 block{};
    dim3 grid{};
    cudaStream_t stream = nullptr;
    double* current = nullptr;
    double* next = nullptr;
    EdgeBuffers edges{};
    bool profiling = false;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;
    Profile timings{};

    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    // Best-effort cleanup also handles partial construction and CUDA failures.
    ~State() noexcept {
        int previous_device = -1;
        const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        if (device >= 0) {
            (void)cudaSetDevice(device);
        }
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
        }
        if (current != nullptr) (void)cudaFree(current);
        if (next != nullptr) (void)cudaFree(next);
        if (edges.send_top != nullptr) (void)cudaFreeHost(edges.send_top);
        if (edges.send_bottom != nullptr) (void)cudaFreeHost(edges.send_bottom);
        if (edges.receive_top != nullptr) (void)cudaFreeHost(edges.receive_top);
        if (edges.receive_bottom != nullptr) (void)cudaFreeHost(edges.receive_bottom);
        if (event_start != nullptr) (void)cudaEventDestroy(event_start);
        if (event_end != nullptr) (void)cudaEventDestroy(event_end);
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
        if (have_previous && previous_device != device) {
            (void)cudaSetDevice(previous_device);
        }
    }
};

namespace {

State& require_state(State* state) {
    if (state == nullptr) {
        throw std::invalid_argument("CUDA state must not be null");
    }
    return *state;
}

State& select_state(State* state) {
    State& selected = require_state(state);
    check_cuda(cudaSetDevice(selected.device), "cudaSetDevice");
    return selected;
}

void start_stage(State& state) {
    if (state.profiling) {
        check_cuda(cudaEventRecord(state.event_start, state.stream),
                   "cudaEventRecord start");
    }
}

void finish_stage(State& state, double& accumulated_seconds) {
    if (state.profiling) {
        check_cuda(cudaEventRecord(state.event_end, state.stream),
                   "cudaEventRecord end");
        check_cuda(cudaEventSynchronize(state.event_end),
                   "cudaEventSynchronize");
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, state.event_start,
                                        state.event_end),
                   "cudaEventElapsedTime");
        accumulated_seconds += static_cast<double>(milliseconds) * 1e-3;
    }
}

void allocate_edge(double*& buffer, std::size_t bytes, int n) {
    check_cuda(cudaMallocHost(reinterpret_cast<void**>(&buffer), bytes),
               "cudaMallocHost halo buffer");
    std::fill_n(buffer, static_cast<std::size_t>(n), 0.0);
}

}  // namespace

State* create(Domain domain, int device, int block_x, int block_y,
              bool profiling) {
    if (domain.n < 4 || domain.rows <= 0 || domain.start < 0 ||
        static_cast<long long>(domain.start) + domain.rows > domain.n) {
        throw std::invalid_argument("Invalid CUDA domain");
    }
    if (device < 0 || block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("Device and CUDA block dimensions are invalid");
    }

    const std::size_t width = static_cast<std::size_t>(domain.n);
    const std::size_t field_cells = checked_product(
        static_cast<std::size_t>(domain.rows) + 2, width,
        "CUDA field element count overflow");
    const std::size_t field_bytes = checked_product(
        field_cells, sizeof(double), "CUDA field byte count overflow");

    check_cuda(cudaSetDevice(device), "cudaSetDevice during create");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    const long long threads = static_cast<long long>(block_x) * block_y;
    if (block_x > properties.maxThreadsDim[0] ||
        block_y > properties.maxThreadsDim[1] ||
        threads > properties.maxThreadsPerBlock) {
        throw std::invalid_argument("CUDA block dimensions exceed device limits");
    }
    if (properties.maxGridSize[0] <= 0 || properties.maxGridSize[1] <= 0) {
        throw std::runtime_error("CUDA device reported invalid grid limits");
    }

    std::unique_ptr<State> state(new State);
    state->domain = domain;
    state->device = device;
    state->row_bytes = checked_product(width, sizeof(double),
                                       "CUDA row byte count overflow");
    state->owned_cells = checked_product(static_cast<std::size_t>(domain.rows),
                                         width, "CUDA owned element count overflow");
    state->block = dim3(static_cast<unsigned int>(block_x),
                        static_cast<unsigned int>(block_y), 1);
    const std::size_t needed_x =
        1 + (width - 3) / static_cast<std::size_t>(block_x);
    const std::size_t needed_y =
        1 + (static_cast<std::size_t>(domain.rows) - 1) /
                static_cast<std::size_t>(block_y);
    state->grid = dim3(
        static_cast<unsigned int>(std::min(
            needed_x, static_cast<std::size_t>(properties.maxGridSize[0]))),
        static_cast<unsigned int>(std::min(
            needed_y, static_cast<std::size_t>(properties.maxGridSize[1]))),
        1);
    state->profiling = profiling;

    check_cuda(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&state->current), field_bytes),
               "cudaMalloc current");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&state->next), field_bytes),
               "cudaMalloc next");
    allocate_edge(state->edges.send_top, state->row_bytes, domain.n);
    allocate_edge(state->edges.send_bottom, state->row_bytes, domain.n);
    allocate_edge(state->edges.receive_top, state->row_bytes, domain.n);
    allocate_edge(state->edges.receive_bottom, state->row_bytes, domain.n);
    if (profiling) {
        check_cuda(cudaEventCreate(&state->event_start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&state->event_end), "cudaEventCreate end");
    }

    // Common initialization uses global coordinates; initialize_grid() is for
    // a full N*N CPU field and must not be given this local allocation.
    std::vector<double> initial(state->owned_cells, 0.0);
    for (int local_row = 0; local_row < domain.rows; ++local_row) {
        const int global_row = domain.start + local_row;
        for (int column = 0; column < domain.n; ++column) {
            initial[static_cast<std::size_t>(local_row) * width + column] =
                heat::initial_value(global_row, column, domain.n);
        }
    }
    check_cuda(cudaMemsetAsync(state->current, 0, field_bytes, state->stream),
               "cudaMemsetAsync current");
    check_cuda(cudaMemsetAsync(state->next, 0, field_bytes, state->stream),
               "cudaMemsetAsync next");
    // Complete initialization before using a synchronous copy from pageable
    // vector storage. Whole-field transfers are outside benchmark profiling.
    check_cuda(cudaStreamSynchronize(state->stream),
               "cudaStreamSynchronize initialization");
    check_cuda(cudaMemcpy(state->current + width, initial.data(),
                          state->owned_cells * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy initial field");
    // A pageable H2D cudaMemcpy may return once staging finishes. It uses the
    // default stream, so explicitly finish that stream before any work on our
    // nonblocking stream can read the initial field.
    check_cuda(cudaStreamSynchronize(nullptr),
               "cudaStreamSynchronize initial upload");
    return state.release();
}

void destroy(State* state) noexcept {
    delete state;
}

void synchronize(State* state) {
    State& selected = select_state(state);
    check_cuda(cudaStreamSynchronize(selected.stream), "cudaStreamSynchronize");
}

EdgeBuffers edge_buffers(State* state) {
    return require_state(state).edges;
}

void pack_edges(State* state) {
    State& selected = select_state(state);
    const std::size_t width = static_cast<std::size_t>(selected.domain.n);
    start_stage(selected);
    check_cuda(cudaMemcpyAsync(selected.edges.send_top,
                               selected.current + width,
                               selected.row_bytes, cudaMemcpyDeviceToHost,
                               selected.stream),
               "cudaMemcpyAsync top edge device to host");
    check_cuda(cudaMemcpyAsync(selected.edges.send_bottom,
                               selected.current +
                                   static_cast<std::size_t>(selected.domain.rows) * width,
                               selected.row_bytes, cudaMemcpyDeviceToHost,
                               selected.stream),
               "cudaMemcpyAsync bottom edge device to host");
    finish_stage(selected, selected.timings.d2h_s);
    if (!selected.profiling) {
        // MPI must not read send buffers before these copies finish. This also
        // completes prior H2D copies before MPI can overwrite receive buffers.
        check_cuda(cudaStreamSynchronize(selected.stream),
                   "cudaStreamSynchronize packed edges");
    }
}

void install_halos(State* state, bool has_up, bool has_down) {
    State& selected = select_state(state);
    if (!has_up && !has_down) {
        return;
    }
    const std::size_t width = static_cast<std::size_t>(selected.domain.n);
    start_stage(selected);
    if (has_up) {
        check_cuda(cudaMemcpyAsync(selected.current, selected.edges.receive_top,
                                   selected.row_bytes, cudaMemcpyHostToDevice,
                                   selected.stream),
                   "cudaMemcpyAsync upper halo host to device");
    }
    if (has_down) {
        check_cuda(cudaMemcpyAsync(
                       selected.current +
                           (static_cast<std::size_t>(selected.domain.rows) + 1) * width,
                       selected.edges.receive_bottom, selected.row_bytes,
                       cudaMemcpyHostToDevice, selected.stream),
                   "cudaMemcpyAsync lower halo host to device");
    }
    // Same-stream ordering puts both copies before the dependent stencil.
    finish_stage(selected, selected.timings.h2d_s);
}

void step_and_swap(State* state) {
    State& selected = select_state(state);
    start_stage(selected);
    heat_stencil<<<selected.grid, selected.block, 0, selected.stream>>>(
        selected.current, selected.next, selected.domain.n,
        selected.domain.rows, selected.domain.start, heat::R);
    check_cuda(cudaGetLastError(), "heat_stencil kernel launch");
    finish_stage(selected, selected.timings.kernel_s);
    std::swap(selected.current, selected.next);
}

std::vector<double> copy_owned_to_host(State* state) {
    State& selected = select_state(state);
    std::vector<double> owned(selected.owned_cells);
    check_cuda(cudaStreamSynchronize(selected.stream),
               "cudaStreamSynchronize final owned field");
    check_cuda(cudaMemcpy(owned.data(), selected.current + selected.domain.n,
                          selected.owned_cells * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy final owned field");
    return owned;
}

Profile profile(State* state) {
    return require_state(state).timings;
}

}  // namespace heat_cuda
