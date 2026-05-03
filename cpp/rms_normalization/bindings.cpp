#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_rms_norm(const float *X, float *Y, size_t b, size_t n, float eps,
                     int grid_dim, int block_dim, cudaStream_t stream);
void launch_rms_norm_warp(const float *X, float *Y, size_t b, size_t n, float eps,
                          int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor rms_norm(torch::Tensor x, double eps) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(1) % 4 == 0, "row length must be divisible by 4");
    auto out = torch::empty_like(x);
    size_t b = x.size(0), n = x.size(1);
    launch_rms_norm(x.data_ptr<float>(), out.data_ptr<float>(), b, n, (float)eps,
                    b, 256, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor rms_norm_warp(torch::Tensor x, double eps) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(1) % 4 == 0, "row length must be divisible by 4");
    auto out = torch::empty_like(x);
    size_t b = x.size(0), n = x.size(1);
    launch_rms_norm_warp(x.data_ptr<float>(), out.data_ptr<float>(), b, n, (float)eps,
                         b, 256, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, py::arg("x"), py::arg("eps") = 1e-6);
    m.def("rms_norm_warp", &rms_norm_warp, py::arg("x"), py::arg("eps") = 1e-6);
}
