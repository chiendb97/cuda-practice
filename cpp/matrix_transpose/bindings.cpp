#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_copy_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_copy_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_transpose_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_transpose_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_transpose_diagonal_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_transpose_diagonal_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_transpose_shared_memory(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);
void launch_transpose_shared_memory_padding(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).dtype() == torch::kInt32, #x " must be int32")

static constexpr int TILE = 16;

// Copy variants: input shape (N, M) row-major, output same shape
static torch::Tensor _copy(torch::Tensor x, void (*fn)(int*, int*, int, int, dim3, dim3, cudaStream_t)) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    int N = x.size(0), M = x.size(1);
    auto out = torch::empty_like(x);
    dim3 block(TILE, TILE);
    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    fn(x.data_ptr<int>(), out.data_ptr<int>(), M, N, grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

// Transpose variants: input shape (N, M), output shape (M, N)
static torch::Tensor _transpose(torch::Tensor x, void (*fn)(int*, int*, int, int, dim3, dim3, cudaStream_t)) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    int N = x.size(0), M = x.size(1);
    auto out = torch::empty({M, N}, x.options());
    dim3 block(TILE, TILE);
    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    fn(x.data_ptr<int>(), out.data_ptr<int>(), M, N, grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

// Diagonal variants need square grid: M == N
static torch::Tensor _transpose_diagonal(torch::Tensor x, void (*fn)(int*, int*, int, int, dim3, dim3, cudaStream_t)) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(0) == x.size(1), "diagonal transpose requires square matrix");
    int N = x.size(0), M = x.size(1);
    auto out = torch::empty({M, N}, x.options());
    dim3 block(TILE, TILE);
    int g = (M + TILE - 1) / TILE;
    dim3 grid(g, g);
    fn(x.data_ptr<int>(), out.data_ptr<int>(), M, N, grid, block, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor copy_row(torch::Tensor x)        { return _copy(x, launch_copy_row); }
torch::Tensor copy_column(torch::Tensor x)     { return _copy(x, launch_copy_column); }
torch::Tensor transpose_row(torch::Tensor x)   { return _transpose(x, launch_transpose_row); }
torch::Tensor transpose_column(torch::Tensor x){ return _transpose(x, launch_transpose_column); }
torch::Tensor transpose_diagonal_row(torch::Tensor x)    { return _transpose_diagonal(x, launch_transpose_diagonal_row); }
torch::Tensor transpose_diagonal_column(torch::Tensor x) { return _transpose_diagonal(x, launch_transpose_diagonal_column); }
torch::Tensor transpose_shared_memory(torch::Tensor x)         { return _transpose(x, launch_transpose_shared_memory); }
torch::Tensor transpose_shared_memory_padding(torch::Tensor x) { return _transpose(x, launch_transpose_shared_memory_padding); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_row", &copy_row);
    m.def("copy_column", &copy_column);
    m.def("transpose_row", &transpose_row);
    m.def("transpose_column", &transpose_column);
    m.def("transpose_diagonal_row", &transpose_diagonal_row);
    m.def("transpose_diagonal_column", &transpose_diagonal_column);
    m.def("transpose_shared_memory", &transpose_shared_memory);
    m.def("transpose_shared_memory_padding", &transpose_shared_memory_padding);
}
