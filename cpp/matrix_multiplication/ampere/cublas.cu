//
// Created by root on 7/7/25.
//

#include <iostream>
#include <random>
#include <ctime>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <cublas_v2.h>

#include "host_utils.cuh"

DEFINE_uint32(M, 512, "M");
DEFINE_uint32(N, 512, "N");
DEFINE_uint32(K, 512, "K");
DEFINE_double(alpha, 1.0, "alpha");
DEFINE_double(beta, 1.0, "beta");

DEFINE_bool(test, false, "test");


void
matrix_multiplication_cublas(const uint32_t M, const uint32_t N, const uint32_t K,
                      const float alpha, half *A, half *B, const float beta, half *C) {
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
}


int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    uint32_t M = FLAGS_M;
    uint32_t N = FLAGS_N;
    uint32_t K = FLAGS_K;
    auto alpha = static_cast<float>(FLAGS_alpha);
    auto beta = static_cast<float>(FLAGS_beta);

    bool test = FLAGS_test;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));


    auto [host_params, device_params] = setup_params<half>(M, N, K, alpha, beta);

    matrix_multiplication_cublas(device_params.M, device_params.N, device_params.K,
                  device_params.alpha, device_params.A, device_params.B,
                  device_params.beta, device_params.C);

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    if (test) {
        std::clock_t time_start = std::clock();
        matrix_multiplication_cpu(host_params.M, host_params.N, host_params.K,
                                  host_params.alpha, host_params.A, host_params.B,
                                  host_params.beta, host_params.C, host_params.D);
        std::clock_t time_end = std::clock();

        double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
        std::cout << "Latency for matrix multiplication on CPU: " << latency_cpu << std::endl;

        if (check_result(device_params.C, host_params.D, M, N, 1e-4)) {
            std::cout << "Result is correct" << std::endl;
        } else {
            std::cout << "Result is incorrect" << std::endl;
        }
    }

    free_params(host_params, device_params);
    return 0;
}
