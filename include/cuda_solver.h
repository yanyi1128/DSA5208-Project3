#ifndef HEAT_CUDA_SOLVER_H
#define HEAT_CUDA_SOLVER_H

#include <vector>

namespace heat_cuda {

// Global grid width, number of owned rows, and first owned global row.
// Device fields contain (rows + 2) * n doubles, with owned local rows 1..rows.
struct Domain {
    int n;
    int rows;
    int start;
};

// Four separate pinned host buffers, each containing n doubles. State owns them.
// MPI may use these pointers only after pack_edges() has returned. Complete all
// MPI requests before install_halos(), and do not modify receive buffers again
// until the next pack_edges() or synchronize() has returned.
struct EdgeBuffers {
    double* send_top;
    double* send_bottom;
    double* receive_top;
    double* receive_bottom;
};

// Optional diagnostic event timings, in seconds. Initial/final whole-field
// transfers are excluded. Profiling synchronizes every measured stage and
// therefore changes execution overhead; use separate runs for scaling timings.
struct Profile {
    double kernel_s = 0.0;
    double d2h_s = 0.0;
    double h2d_s = 0.0;
};

struct State;

// Select device, allocate/initialize a normalized field using heat_common.h,
// and validate the block shape against this device's limits. Initialization is
// complete on return. CUDA failures throw std::runtime_error; invalid arguments
// and allocation-size overflows throw standard argument/overflow exceptions.
State* create(Domain domain, int device, int block_x, int block_y, bool profiling);
void destroy(State* state) noexcept;
void synchronize(State* state);

EdgeBuffers edge_buffers(State* state);
void pack_edges(State* state);
// Enqueue copies for real neighbors only, ordered before the next update.
void install_halos(State* state, bool has_up, bool has_down);
// One explicit update; physical boundary cells remain zero in both buffers.
void step_and_swap(State* state);
// Wait for a copy of owned rows, excluding both halos; suitable for MPI_Gatherv.
std::vector<double> copy_owned_to_host(State* state);
Profile profile(State* state);

}  // namespace heat_cuda

#endif
