#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_conv_1d(const float *A, const float *B, float *out, size_t n, int k,
                    int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor conv_1d(torch::Tensor x, torch::Tensor w) {
    CHECK_INPUT(x); CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    TORCH_CHECK(w.dim() == 1, "w must be 1D");
    TORCH_CHECK(w.size(0) % 2 == 1, "kernel size must be odd");
    auto out = torch::empty_like(x);
    size_t n = x.size(0);
    int k = w.size(0);
    int block = 256;
    int grid = (n + block - 1) / block;
    launch_conv_1d(x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(),
                   n, k, grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_1d", &conv_1d);
}
