FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ONEDNN_TAG=v3.9.1
ARG ONEDNN_PREFIX=/opt/onednn-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libjpeg-dev \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --depth 1 --branch ${ONEDNN_TAG} https://github.com/oneapi-src/oneDNN.git

WORKDIR /tmp/oneDNN
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${ONEDNN_PREFIX} \
    -DONEDNN_BUILD_TESTS=OFF \
    -DONEDNN_BUILD_EXAMPLES=OFF \
    -DONEDNN_CPU_RUNTIME=OMP \
    -DONEDNN_GPU_RUNTIME=NONE \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build

WORKDIR /workspace

ENV DNNL_ROOT=${ONEDNN_PREFIX}
ENV CMAKE_PREFIX_PATH=${ONEDNN_PREFIX}
ENV LD_LIBRARY_PATH=${ONEDNN_PREFIX}/lib:${ONEDNN_PREFIX}/lib64:${LD_LIBRARY_PATH}

# Example inside the container:
#   cmake -S . -B build-probe -DDNNL_ROOT=${DNNL_ROOT} -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
#   cmake --build build-probe --target test_onednn_conv_parity_probe -j"$(nproc)"
#   ./build-probe/test_onednn_conv_parity_probe
