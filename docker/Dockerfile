ARG BASE_IMAGE=harbor.infra.zalo.services/nvcr_proxy_cache/nvidia/tritonserver
ARG BASE_TAG=24.11-py3

FROM ${BASE_IMAGE}:${BASE_TAG} as base

ENV http_proxy=http://10.60.28.99:81
ENV https_proxy=http://10.60.28.99:81

RUN apt-get update && apt-get install -y --no-install-recommends python-is-python3 rapidjson-dev ccache git-lfs tzdata net-tools

# Remove previous TRT installation
# We didn't remove libnvinfer* here because tritonserver depends on the pre-installed libraries.
RUN apt-get remove --purge -y tensorrt*
RUN pip uninstall -y tensorrt

FROM base as dev

ENV SHINIT_FILE=${BASH_ENV}

# Download & install internal TRT release
COPY docker/common/install_tensorrt.sh /tmp/
RUN bash /tmp/install_tensorrt.sh && \
    rm -f /tmp/install_tensorrt.sh
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

# Install requirements
COPY docker/common/requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt --extra-index-url https://pypi.ngc.nvidia.com && \
    rm -f /tmp/requirements.txt

FROM dev as vscode_server_builder

RUN wget https://github.com/coder/code-server/releases/download/v4.96.4/code-server_4.96.4_amd64.deb && \
    dpkg -i code-server_4.96.4_amd64.deb && \
    rm -f code-server_4.96.4_amd64.deb

RUN code-server --install-extension ms-python.python

RUN wget https://github.com/microsoft/vscode-cpptools/releases/download/v1.23.6/cpptools-linux-x64.vsix && \
    code-server --install-extension cpptools-linux-x64.vsix && \
    rm -f cpptools-linux-x64.vsix

COPY docker/common/NVIDIA.nsight-vscode-edition-2024.1.34572442.vsix NVIDIA.nsight-vscode-edition.vsix
RUN code-server --install-extension NVIDIA.nsight-vscode-edition.vsix && \
    rm -f NVIDIA.nsight-vscode-edition.vsix

FROM vscode_server_builder as final

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

ENV OPAL_PREFIX="/opt/hpcx/ompi"
ENV PATH="/opt/hpcx/ompi/bin:${PATH}"
ENV PATH="/opt/hpcx/ompi/include:${PATH}"
ENV LD_LIBRARY_PATH="/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}"

ENV http_proxy=""
ENV https_proxy=""