#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

void launch_matmul_hopper_opt1(uint32_t M, uint32_t N, uint32_t K,
                               float alpha, half *A, half *B, float beta, half *C, half *D,
                               CUtensorMap *A_map, CUtensorMap *B_map, cudaStream_t stream);
void launch_matmul_hopper_opt2(uint32_t M, uint32_t N, uint32_t K,
                               float alpha, half *A, half *B, float beta, half *C, half *D,
                               CUtensorMap *A_map, CUtensorMap *B_map, cudaStream_t stream);
void launch_matmul_hopper_opt3(uint32_t M, uint32_t N, uint32_t K,
                               float alpha, half *A, half *B, float beta, half *C, half *D,
                               CUtensorMap *A_map, CUtensorMap *B_map, cudaStream_t stream);

#define CHECK_HALF(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).dtype() == torch::kHalf, #x " must be float16")

using KernelFn = void (*)(uint32_t, uint32_t, uint32_t, float, half *, half *, float, half *, half *,
                          CUtensorMap *, CUtensorMap *, cudaStream_t);

static torch::Tensor _matmul_hopper(KernelFn fn, double alpha, torch::Tensor A,
                                    torch::Tensor B, double beta, torch::Tensor C) {
    CHECK_HALF(A); CHECK_HALF(B); CHECK_HALF(C);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);

    auto D = torch::empty({M, N}, A.options());

    auto map_opts = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
    auto A_map_buf = torch::empty({(int64_t)sizeof(CUtensorMap)}, map_opts);
    auto B_map_buf = torch::empty({(int64_t)sizeof(CUtensorMap)}, map_opts);

    fn((uint32_t)M, (uint32_t)N, (uint32_t)K, (float)alpha,
       reinterpret_cast<half *>(A.data_ptr<at::Half>()),
       reinterpret_cast<half *>(B.data_ptr<at::Half>()),
       (float)beta,
       reinterpret_cast<half *>(C.data_ptr<at::Half>()),
       reinterpret_cast<half *>(D.data_ptr<at::Half>()),
       reinterpret_cast<CUtensorMap *>(A_map_buf.data_ptr()),
       reinterpret_cast<CUtensorMap *>(B_map_buf.data_ptr()),
       at::cuda::getCurrentCUDAStream());
    return D;
}

torch::Tensor matmul_hopper_opt1(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_hopper(launch_matmul_hopper_opt1, alpha, A, B, beta, C);
}
torch::Tensor matmul_hopper_opt2(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_hopper(launch_matmul_hopper_opt2, alpha, A, B, beta, C);
}
torch::Tensor matmul_hopper_opt3(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_hopper(launch_matmul_hopper_opt3, alpha, A, B, beta, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_hopper_opt1", &matmul_hopper_opt1);
    m.def("matmul_hopper_opt2", &matmul_hopper_opt2);
    m.def("matmul_hopper_opt3", &matmul_hopper_opt3);
}
