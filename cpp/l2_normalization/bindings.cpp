#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_l2_norm(const float *X, float *Y, size_t b, size_t n, float eps,
                    int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor l2_norm(torch::Tensor x, double eps) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(1) % 4 == 0, "row length must be divisible by 4");
    auto out = torch::empty_like(x);
    size_t b = x.size(0), n = x.size(1);
    launch_l2_norm(x.data_ptr<float>(), out.data_ptr<float>(), b, n, (float)eps,
                   b, 256, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2_norm", &l2_norm, py::arg("x"), py::arg("eps") = 1e-12);
}
