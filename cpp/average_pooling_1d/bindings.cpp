#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_average_pooling_1d(const float *X, float *Y, int kernel_size, int padding, int stride,
                               int H, int H_out, int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor avg_pool_1d(torch::Tensor x, int64_t kernel_size, int64_t padding, int64_t stride) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    int H = x.size(0);
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    auto out = torch::empty({H_out}, x.options());
    int block = 256;
    int grid = (H_out + block - 1) / block;
    launch_average_pooling_1d(x.data_ptr<float>(), out.data_ptr<float>(),
                              kernel_size, padding, stride, H, H_out,
                              grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool_1d", &avg_pool_1d);
}
