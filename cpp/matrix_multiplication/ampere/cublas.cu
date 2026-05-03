//
// Created by root on 7/7/25.
//

#include <cublas_v2.h>
#include "../../cuda_common.cuh"
#include <cuda_fp16.h>

#define CHECK_CUBLAS_ERROR(val) check_cublas_error((val), #val, __FILE__, __LINE__)
inline void check_cublas_error(cublasStatus_t err, const char *func, const char *file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << "\n"
                  << "CUBLAS_STATUS_ERROR " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void launch_matmul_cublas(const uint32_t M, const uint32_t N, const uint32_t K,
                          const float alpha, half *A, half *B, const float beta, half *C,
                          cudaStream_t stream) {
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    CHECK_CUBLAS_ERROR(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DFALT_TENSOR_OP));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}
