#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_hello_cuda(int nx, int ny, int block_x, int block_y, cudaStream_t stream);

void hello(int nx, int ny, int block_x, int block_y) {
    launch_hello_cuda(nx, ny, block_x, block_y, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hello", &hello,
          py::arg("nx") = 16, py::arg("ny") = 16,
          py::arg("block_x") = 4, py::arg("block_y") = 4);
}
