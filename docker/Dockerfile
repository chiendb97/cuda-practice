ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_TAG=25.05-py3

FROM ${BASE_IMAGE}:${BASE_TAG} AS base

ENV BASH_ENV=${BASH_ENV:-/etc/bash.bashrc}
ENV ENV=${ENV:-/etc/shinit_v2}

ENV SHINIT_FILE=${BASH_ENV}

ARG http_proxy
ARG https_proxy

RUN apt-get update && apt-get install -y --no-install-recommends \
    rapidjson-dev ccache gdb git-lfs tzdata net-tools \
    clang lld llvm libclang-rt-dev libffi-dev libibverbs-dev libnuma1 libnuma-dev \
    python3-dev python3-pip python-is-python3 wget pigz libgflags-dev

# Remove previous TRT installation
# We didn't remove libnvinfer* here because tritonserver depends on the pre-installed libraries.
RUN apt-get remove --purge -y tensorrt*
RUN pip uninstall -y tensorrt

FROM base as dev

# Todo: Add proxy
# Download & install internal TRT release
ARG TRT_VER
ARG CUDA_VER
ARG CUDNN_VER
ARG NCCL_VER
ARG CUBLAS_VER
COPY docker/common/install_tensorrt.sh /tmp/
RUN bash /tmp/install_tensorrt.sh \
    --TRT_VER=${TRT_VER} \
    --CUDA_VER=${CUDA_VER} \
    --CUDNN_VER=${CUDNN_VER} \
    --NCCL_VER=${NCCL_VER} \
    --CUBLAS_VER=${CUBLAS_VER} && \
    rm -f /tmp/install_tensorrt.sh \

ENV LD_LIBRARY_PATH="/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}"
ENV TRT_ROOT=/usr/local/tensorrt

# Install latest Polygraphy
COPY docker/common/install_polygraphy.sh /tmp/
RUN bash /tmp/install_polygraphy.sh && \
    rm -f /tmp/install_polygraphy.sh

# CMake
COPY docker/common/install_cmake.sh /tmp/
RUN bash /tmp/install_cmake.sh && \
    rm -f /tmp/install_cmake.sh
ENV PATH="/usr/local/cmake/bin:${PATH}"

# Install mpi4py
COPY docker/common/install_mpi4py.sh /tmp/
RUN bash /tmp/install_mpi4py.sh && \
    rm -f /tmp/install_mpi4py.sh

# `pypi` for x86_64 arch and `src_cxx11_abi` for aarch64 arch
ARG TORCH_INSTALL_TYPE="pypi"
COPY docker/common/install_pytorch.sh /tmp/
RUN bash /tmp/install_pytorch.sh $TORCH_INSTALL_TYPE && \
    rm -f /tmp/install_pytorch.sh

COPY docker/common/install_triton.sh /tmp/
RUN bash /tmp/install_triton.sh && \
    rm -f /tmp/install_triton.sh

COPY docker/common/install_cutlass_dsl.sh /tmp/
RUN bash /tmp/install_cutlass_dsl.sh && \
    rm -f /tmp/install_cutlass_dsl.sh

# Install requirements
COPY docker/common/requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt --extra-index-url https://pypi.ngc.nvidia.com && \
    rm -f /tmp/requirements.txt

FROM dev as vscode_server_builder

RUN wget https://github.com/coder/code-server/releases/download/v4.101.2/code-server_4.101.2_amd64.deb && \
    dpkg -i code-server_4.101.2_amd64.deb && \
    rm -f code-server_4.101.2_amd64.deb

# Install vscode extension ms-python.python
# Install vscode extension cpptools
# Install vscode extension nsight-vscode-edition

FROM vscode_server_builder as final

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

ENV OPAL_PREFIX="/opt/hpcx/ompi"
ENV PATH="/opt/hpcx/ompi/bin:${PATH}"
ENV PATH="/opt/hpcx/ompi/include:${PATH}"
ENV LD_LIBRARY_PATH="/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}"