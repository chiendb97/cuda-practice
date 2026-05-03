#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_reduce_neighbored_pair(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_interleaved_pair(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_neighbored_pair_less(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_loop_unrolling(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_warp_unrolling(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_complete_unrolling(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_template_function(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_template_function_shared_memory(int*, int*, int, int, int, cudaStream_t);
void launch_reduce_warp_shuffle_instruction(int*, int*, int, int, int, cudaStream_t);

#define CHECK_INPUT(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).dtype() == torch::kInt32, #x " must be int32")

using LaunchFn = void(*)(int*, int*, int, int, int, cudaStream_t);

// Variants where each block processes blockDim.x elements (no 8x unroll)
static torch::Tensor _reduce_basic(torch::Tensor x, LaunchFn fn) {
    CHECK_INPUT(x);
    auto a = x.clone();
    int size = a.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    auto s = torch::empty({grid}, a.options());
    fn(a.data_ptr<int>(), s.data_ptr<int>(), size, grid, block,
       at::cuda::getCurrentCUDAStream());
    return s.sum();
}

// Variants where each block processes blockDim.x * 8 elements (8x unroll)
static torch::Tensor _reduce_unrolled(torch::Tensor x, LaunchFn fn) {
    CHECK_INPUT(x);
    auto a = x.clone();
    int size = a.numel();
    int block = 256;
    int grid = (size + block * 8 - 1) / (block * 8);
    auto s = torch::empty({grid}, a.options());
    fn(a.data_ptr<int>(), s.data_ptr<int>(), size, grid, block,
       at::cuda::getCurrentCUDAStream());
    return s.sum();
}

torch::Tensor reduce_neighbored_pair(torch::Tensor x)      { return _reduce_basic(x, launch_reduce_neighbored_pair); }
torch::Tensor reduce_interleaved_pair(torch::Tensor x)     { return _reduce_basic(x, launch_reduce_interleaved_pair); }
torch::Tensor reduce_neighbored_pair_less(torch::Tensor x) { return _reduce_basic(x, launch_reduce_neighbored_pair_less); }

// warp_shuffle_instruction reads smem[0..31] but only allocates smem[0..warps-1].
// Run with block=1024 so the 32-warp read is fully covered.
torch::Tensor reduce_warp_shuffle_instruction(torch::Tensor x) {
    CHECK_INPUT(x);
    auto a = x.clone();
    int size = a.numel();
    int block = 1024;
    int grid = (size + block - 1) / block;
    auto s = torch::empty({grid}, a.options());
    launch_reduce_warp_shuffle_instruction(a.data_ptr<int>(), s.data_ptr<int>(),
        size, grid, block, at::cuda::getCurrentCUDAStream());
    return s.sum();
}
torch::Tensor reduce_loop_unrolling(torch::Tensor x)    { return _reduce_unrolled(x, launch_reduce_loop_unrolling); }
torch::Tensor reduce_warp_unrolling(torch::Tensor x)    { return _reduce_unrolled(x, launch_reduce_warp_unrolling); }
torch::Tensor reduce_complete_unrolling(torch::Tensor x){ return _reduce_unrolled(x, launch_reduce_complete_unrolling); }
torch::Tensor reduce_template_function(torch::Tensor x) { return _reduce_unrolled(x, launch_reduce_template_function); }
torch::Tensor reduce_template_function_shared_memory(torch::Tensor x) { return _reduce_unrolled(x, launch_reduce_template_function_shared_memory); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_neighbored_pair", &reduce_neighbored_pair);
    m.def("reduce_interleaved_pair", &reduce_interleaved_pair);
    m.def("reduce_neighbored_pair_less", &reduce_neighbored_pair_less);
    m.def("reduce_loop_unrolling", &reduce_loop_unrolling);
    m.def("reduce_warp_unrolling", &reduce_warp_unrolling);
    m.def("reduce_complete_unrolling", &reduce_complete_unrolling);
    m.def("reduce_template_function", &reduce_template_function);
    m.def("reduce_template_function_shared_memory", &reduce_template_function_shared_memory);
    m.def("reduce_warp_shuffle_instruction", &reduce_warp_shuffle_instruction);
}
