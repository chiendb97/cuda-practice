#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

void launch_matmul_naive(uint32_t M, uint32_t N, uint32_t K,
                         float alpha, half *A, half *B, float beta, half *C, half *D,
                         cudaStream_t stream);
void launch_matmul_opt1(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt2(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt3(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt4(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt5(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt6(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt7(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_opt8(uint32_t M, uint32_t N, uint32_t K,
                        float alpha, half *A, half *B, float beta, half *C, half *D,
                        cudaStream_t stream);
void launch_matmul_cublas(uint32_t M, uint32_t N, uint32_t K,
                          float alpha, half *A, half *B, float beta, half *C,
                          cudaStream_t stream);

#define CHECK_HALF(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be CUDA"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK((x).dtype() == torch::kHalf, #x " must be float16")

using KernelFn = void (*)(uint32_t, uint32_t, uint32_t, float, half *, half *, float, half *, half *, cudaStream_t);

static torch::Tensor _matmul_ampere(KernelFn fn, double alpha, torch::Tensor A,
                                    torch::Tensor B, double beta, torch::Tensor C) {
    CHECK_HALF(A); CHECK_HALF(B); CHECK_HALF(C);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);

    auto D = torch::empty({M, N}, A.options());
    fn((uint32_t)M, (uint32_t)N, (uint32_t)K, (float)alpha,
       reinterpret_cast<half *>(A.data_ptr<at::Half>()),
       reinterpret_cast<half *>(B.data_ptr<at::Half>()),
       (float)beta,
       reinterpret_cast<half *>(C.data_ptr<at::Half>()),
       reinterpret_cast<half *>(D.data_ptr<at::Half>()),
       at::cuda::getCurrentCUDAStream());
    return D;
}

torch::Tensor matmul_ampere_naive(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_naive, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt1(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt1, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt2(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt2, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt3(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt3, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt4(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt4, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt5(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt5, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt6(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt6, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt7(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt7, alpha, A, B, beta, C);
}
torch::Tensor matmul_ampere_opt8(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    return _matmul_ampere(launch_matmul_opt8, alpha, A, B, beta, C);
}

torch::Tensor matmul_ampere_cublas(double alpha, torch::Tensor A, torch::Tensor B, double beta, torch::Tensor C) {
    CHECK_HALF(A); CHECK_HALF(B); CHECK_HALF(C);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    auto out = C.clone();
    launch_matmul_cublas((uint32_t)M, (uint32_t)N, (uint32_t)K, (float)alpha,
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        (float)beta,
        reinterpret_cast<half *>(out.data_ptr<at::Half>()),
        at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_ampere_naive", &matmul_ampere_naive);
    m.def("matmul_ampere_opt1", &matmul_ampere_opt1);
    m.def("matmul_ampere_opt2", &matmul_ampere_opt2);
    m.def("matmul_ampere_opt3", &matmul_ampere_opt3);
    m.def("matmul_ampere_opt4", &matmul_ampere_opt4);
    m.def("matmul_ampere_opt5", &matmul_ampere_opt5);
    m.def("matmul_ampere_opt6", &matmul_ampere_opt6);
    m.def("matmul_ampere_opt7", &matmul_ampere_opt7);
    m.def("matmul_ampere_opt8", &matmul_ampere_opt8);
    m.def("matmul_ampere_cublas", &matmul_ampere_cublas);
}
