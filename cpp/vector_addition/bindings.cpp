#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_vector_add(const float *a, const float *b, float *out, size_t n,
                       int grid_dim, int block_dim, cudaStream_t stream);
void launch_vector_add_vec4(const float *a, const float *b, float *out, size_t n,
                            int grid_dim, int block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a); CHECK_INPUT(b);
    auto out = torch::empty_like(a);
    int n = a.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                      n, grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor vector_add_vec4(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a); CHECK_INPUT(b);
    TORCH_CHECK(a.numel() % 4 == 0, "numel must be divisible by 4");
    auto out = torch::empty_like(a);
    int n = a.numel();
    int block = 256;
    int grid = (n / 4 + block - 1) / block;
    launch_vector_add_vec4(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                           n, grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add);
    m.def("vector_add_vec4", &vector_add_vec4);
}
