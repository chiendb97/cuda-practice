#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_relu(const float *X, float *Y, size_t n, size_t m,
                 int grid_dim, int block_dim, cudaStream_t stream);
void launch_relu_opt(const float *X, float *Y, size_t n, size_t m,
                     int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor relu_baseline(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK((x.size(0) * x.size(1)) % 4 == 0, "numel must be divisible by 4");
    auto out = torch::empty_like(x);
    size_t n = x.size(0), m = x.size(1);
    int block = 256;
    int grid = (n * m / 4 + block - 1) / block;
    launch_relu(x.data_ptr<float>(), out.data_ptr<float>(), n, m,
                grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor relu_optimize(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK((x.size(0) * x.size(1)) % 4 == 0, "numel must be divisible by 4");
    auto out = torch::empty_like(x);
    size_t n = x.size(0), m = x.size(1);
    int block = 256;
    int grid = (n * m / 4 + block - 1) / block;
    launch_relu_opt(x.data_ptr<float>(), out.data_ptr<float>(), n, m,
                    grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_baseline", &relu_baseline);
    m.def("relu_optimize", &relu_optimize);
}
