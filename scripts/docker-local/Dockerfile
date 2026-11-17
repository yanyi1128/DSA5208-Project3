# Local GPU development only. Results from this image are not Atlas/Vanda runs.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-nvcc-12-1 cuda-cudart-dev-12-1 \
    build-essential openmpi-bin libopenmpi-dev python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 runner
ENV CUDA_ARCH=sm_86
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER runner
WORKDIR /project
