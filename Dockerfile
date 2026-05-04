# SimGrid + MPI build environment for the CMSC 714 topology study.
# Provides a consistent Linux/SimGrid setup so teammates on macOS or Windows
# (and native Linux folks who want consistency) build and benchmark from the
# same toolchain.
#
# Typical use (from the repo root):
#
#   make docker-build     # one-time, ~10 min on first build
#   make docker-shell     # interactive shell, repo mounted at /work
#   make docker-make      # one-shot: run `make` inside the container
#
# Or directly:
#
#   docker build -t cmsc714 .
#   docker run -it --rm -v "$PWD:/work" -w /work cmsc714 bash

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Build tools + SimGrid prereqs (Boost) + FFTW (needed by src/2d_fft_mpi.cpp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ca-certificates \
        wget \
        pkg-config \
        python3 \
        libboost-dev \
        libboost-context-dev \
        libboost-system-dev \
        libfftw3-dev \
        libfftw3-mpi-dev \
    && rm -rf /var/lib/apt/lists/*

# SimGrid is not packaged in Ubuntu 22.04 — build from source. Version pinned
# to match the link in README.md.
ARG SIMGRID_VERSION=4.1
RUN cd /tmp \
    && wget -q "https://github.com/simgrid/simgrid/releases/download/v${SIMGRID_VERSION}/simgrid-${SIMGRID_VERSION}.tar.gz" \
    && tar xf "simgrid-${SIMGRID_VERSION}.tar.gz" \
    && cd "simgrid-${SIMGRID_VERSION}" \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -Denable_documentation=OFF \
        -Denable_python=OFF \
        . \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && cd / \
    && rm -rf "/tmp/simgrid-${SIMGRID_VERSION}"* \
    && ldconfig

WORKDIR /work
CMD ["bash"]
