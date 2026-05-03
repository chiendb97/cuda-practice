#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_prefix_multiple(const float *X, float *Y, int n,
                            int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")

torch::Tensor prefix_multiple(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    launch_prefix_multiple(x.data_ptr<float>(), out.data_ptr<float>(), n,
                           grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefix_multiple", &prefix_multiple);
}
