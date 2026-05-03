#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_prefix_sum_cub(const float *X, float *Y, int n,
                           int grid_dim, int block_dim, cudaStream_t stream);
void launch_prefix_sum_opt(const float *X, float *Y, int n,
                           int *flags, float *scan_value, int *block_counter,
                           int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")

torch::Tensor prefix_sum_cub(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    launch_prefix_sum_cub(x.data_ptr<float>(), out.data_ptr<float>(), n,
                          grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor prefix_sum_opt(torch::Tensor x) {
    CHECK_INPUT(x);
    int n = x.numel();
    constexpr int block = 256;
    constexpr int items_per_thread = 4;
    TORCH_CHECK(n % (block * items_per_thread) == 0,
                "numel must be divisible by block_dim * items_per_thread = 1024");
    int grid = n / (block * items_per_thread);
    auto out = torch::empty_like(x);

    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto opts_flt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto flags = torch::zeros({grid + 1}, opts_int);
    auto scan_value = torch::zeros({grid + 1}, opts_flt);
    auto block_counter = torch::zeros({1}, opts_int);
    flags.index_put_({0}, 1);

    launch_prefix_sum_opt(x.data_ptr<float>(), out.data_ptr<float>(), n,
                          flags.data_ptr<int>(), scan_value.data_ptr<float>(),
                          block_counter.data_ptr<int>(),
                          grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefix_sum_cub", &prefix_sum_cub);
    m.def("prefix_sum_opt", &prefix_sum_opt);
}
