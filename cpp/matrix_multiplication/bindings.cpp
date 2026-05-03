#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

void launch_matmul_cuda_core(int M, int N, int K,
                             const float *A, const float *B, float *C,
                             cudaStream_t stream);
void launch_matmul_cuda_core_opt(int M, int N, int K,
                                 const float *A, const float *B, float *C,
                                 cudaStream_t stream);
void launch_matmul_tensor_core(int M, int N, int K,
                               const half *A, const half *B, float *C,
                               cudaStream_t stream);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")

// Column-major matmul: kernel computes C = A @ B (where A is M×K, B is K×N).
// Internally A is column-major M×K (= A.t() in row-major), and B is row-major
// K×N for tensor_core / N×K column-major (= B as-is in row-major) for cuda_core
// — see kernel for details. The output C_col is column-major M×N (= N×M
// row-major), which we transpose back to row-major M×N.

torch::Tensor matmul_cuda_core(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    TORCH_CHECK(A.dtype() == torch::kFloat32);
    TORCH_CHECK(B.dtype() == torch::kFloat32);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);

    auto A_col = A.t().contiguous();                  // K×M row-major == M×K col-major
    auto B_in  = B.contiguous();                      // K×N row-major (kernel reads as N×K col-major)
    auto C_col = torch::empty({N, M}, A.options());   // N×M row-major == M×N col-major

    launch_matmul_cuda_core(M, N, K,
        A_col.data_ptr<float>(), B_in.data_ptr<float>(), C_col.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream());
    return C_col.t().contiguous();
}

torch::Tensor matmul_cuda_core_opt(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    TORCH_CHECK(A.dtype() == torch::kFloat32);
    TORCH_CHECK(B.dtype() == torch::kFloat32);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(M % 32 == 0 && N % 32 == 0 && K % 32 == 0,
                "Optimized variant requires M, N, K divisible by 32");

    auto A_col = A.t().contiguous();
    auto B_in  = B.contiguous();
    auto C_col = torch::empty({N, M}, A.options());

    launch_matmul_cuda_core_opt(M, N, K,
        A_col.data_ptr<float>(), B_in.data_ptr<float>(), C_col.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream());
    return C_col.t().contiguous();
}

torch::Tensor matmul_tensor_core(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    TORCH_CHECK(A.dtype() == torch::kHalf);
    TORCH_CHECK(B.dtype() == torch::kHalf);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);

    auto A_col = A.t().contiguous();
    auto B_in  = B.contiguous();
    auto C_col = torch::empty({N, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    launch_matmul_tensor_core(M, N, K,
        reinterpret_cast<const half*>(A_col.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_in.data_ptr<at::Half>()),
        C_col.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream());
    return C_col.t().contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_core", &matmul_cuda_core);
    m.def("matmul_cuda_core_opt", &matmul_cuda_core_opt);
    m.def("matmul_tensor_core", &matmul_tensor_core);
}
