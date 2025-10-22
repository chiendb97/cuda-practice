//
// Created by root on 10/20/25.
//

#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <cuda_fp16.h>
#include <gflags/gflags.h>

#include "host_utils.cuh"
#include "device_utils.cuh"

DEFINE_uint32(M, 512, "M");
DEFINE_uint32(N, 512, "N");
DEFINE_uint32(K, 512, "K");
DEFINE_double(alpha, 1.0, "alpha");
DEFINE_double(beta, 1.0, "beta");

DEFINE_bool(test, false, "test");
DEFINE_uint32(num_warmups, 0, "num_warmups");
DEFINE_uint32(num_repeats, 1, "num_repeats");


template <uint32_t BM, uint32_t BN, uint32_t BK,
          uint32_t WM, uint32_t WN, uint32_t WK,
          uint32_t MMA_M, uint32_t MMA_N, uint32_t MMA_K,
          uint32_t NUM_THREADS, uint32_t STAGES>
__global__ void
matrix_multiplication(const uint32_t M, const uint32_t N, const uint32_t K,
                      const float alpha, half* A, half* B, const float beta, half* C, half* D)
{
    const uint32_t A_stride = K;
    const uint32_t B_stride = N;
    const uint32_t CD_stride = N;

    constexpr uint32_t SWIZZLE_BITS_A = int_log2(BK / 8);
    constexpr uint32_t SWIZZLE_BITS_B = int_log2(BN / 8);

    constexpr uint32_t mma_tiles_per_warp_m = WM / MMA_M;
    constexpr uint32_t mma_tiles_per_warp_n = WN / MMA_N;
    constexpr uint32_t mma_tiles_per_warp_k = WK / MMA_K;

    constexpr uint32_t VEC_SIZE = 16 / sizeof(half);

    const uint32_t num_block_tiles_k = K / BK;

    const uint32_t block_m = blockIdx.y;
    const uint32_t block_n = blockIdx.x;

    const uint32_t warp_m = threadIdx.y;
    const uint32_t warp_n = threadIdx.x / 32;

    // Todo: Cast shared memory to SharedStorage
    extern __shared__ half shared_memory[];
    half* A_smem = shared_memory;
    half* B_smem = &shared_memory[STAGES * BM * BK];

    uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];
    uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
    uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

    auto (&acc_register_)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] =
        reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);
    auto (&A_register_)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4] =
        reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]>(A_register);
    auto (&B_register_)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2] =
        reinterpret_cast<half(&)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]>(B_register);

#pragma unroll
    for (uint32_t mma_m = 0; mma_m < mma_tiles_per_warp_m; ++mma_m)
    {
#pragma unroll
        for (uint32_t mma_n = 0; mma_n < mma_tiles_per_warp_n; ++mma_n)
        {
            acc_register_[mma_m][mma_n][0] = 0;
            acc_register_[mma_m][mma_n][1] = 0;
            acc_register_[mma_m][mma_n][2] = 0;
            acc_register_[mma_m][mma_n][3] = 0;
        }
    }

    //////////////
    // mainloop //
    //////////////

    // Prefetch the (STAGES - 1) block tile of A, B into shared memory
#pragma unroll
    for (uint32_t block_k = 0; block_k < STAGES - 1; ++block_k)
    {
        tiled_mem_cpy_async_swizzle_a<BM, NUM_THREADS, VEC_SIZE>(
            reinterpret_cast<float4*>(A + block_m * BM * A_stride + block_k * BK),
            reinterpret_cast<float4*>(A_smem + block_k * BM * BK),
            A_stride);

        tiled_mem_cpy_async_swizzle<BK, BN, NUM_THREADS, VEC_SIZE, SWIZZLE_BITS_B>(
            reinterpret_cast<float4*>(B + block_k * BK * B_stride + block_n * BN),
            reinterpret_cast<float4*>(B_smem + block_k * BK * BN),
            B_stride);

        cp_async_commit_group();
    }

    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    uint32_t consumer_block = 0;
    uint32_t producer_block = STAGES - 1;

    for (uint32_t block_k = STAGES - 1; block_k < num_block_tiles_k; ++block_k)
    {
        tiled_mem_cpy_async_swizzle_a<BM, NUM_THREADS, VEC_SIZE>(
            reinterpret_cast<float4*>(A + block_m * BM * A_stride + block_k * BK),
            reinterpret_cast<float4*>(A_smem + producer_block * BM * BK),
            A_stride);

        tiled_mem_cpy_async_swizzle<BK, BN, NUM_THREADS, VEC_SIZE, SWIZZLE_BITS_B>(
            reinterpret_cast<float4*>(B + block_k * BK * B_stride + block_n * BN),
            reinterpret_cast<float4*>(B_smem + producer_block * BK * BN),
            B_stride);

        cp_async_commit_group();
        producer_block = producer_block < STAGES - 1 ? producer_block + 1 : 0;

        const half* A_warp_smem = A_smem + consumer_block * BM * BK + warp_m * WM * BK;
        const half* B_warp_smem = B_smem + consumer_block * BK * BN + warp_n * WN;

        ldmatrix_a<mma_tiles_per_warp_m, mma_tiles_per_warp_k, BK>(A_warp_smem, A_register_);
        ldmatrix_b<mma_tiles_per_warp_k, mma_tiles_per_warp_n, BN>(B_warp_smem, B_register_);

        // Outer product between mma tiles
#pragma unroll
        for (uint32_t mma_k = 0; mma_k < mma_tiles_per_warp_k; ++mma_k)
        {
#pragma unroll
            for (uint32_t mma_n = 0; mma_n < mma_tiles_per_warp_n; ++mma_n)
            {
#pragma unroll
                for (uint32_t mma_m = 0; mma_m < mma_tiles_per_warp_m; ++mma_m)
                {
                    asm volatile (
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                        "{%0, %1}, "
                        "{%2, %3}, "
                        "{%4}, "
                        "{%5, %6};"
                        : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
                        : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
                        "r"(B_register[mma_k][mma_n]),
                        "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
                    );
                }
            }
        }

        consumer_block = consumer_block < STAGES - 1 ? consumer_block + 1 : 0;

        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
    }

    if constexpr (STAGES > 2)
    {
        cp_async_wait_group<0>();
        __syncthreads();
    }

#pragma unroll
    for (uint32_t block_k = 0; block_k < STAGES - 1; ++block_k)
    {
        const half* A_warp_smem = A_smem + consumer_block * BM * BK + warp_m * WM * BK;
        const half* B_warp_smem = B_smem + consumer_block * BK * BN + warp_n * WN;

        ldmatrix_a<mma_tiles_per_warp_m, mma_tiles_per_warp_k, BK>(A_warp_smem, A_register_);
        ldmatrix_b<mma_tiles_per_warp_k, mma_tiles_per_warp_n, BN>(B_warp_smem, B_register_);

        // Outer product between mma tiles
#pragma unroll
        for (uint32_t mma_k = 0; mma_k < mma_tiles_per_warp_k; ++mma_k)
        {
#pragma unroll
            for (uint32_t mma_n = 0; mma_n < mma_tiles_per_warp_n; ++mma_n)
            {
#pragma unroll
                for (uint32_t mma_m = 0; mma_m < mma_tiles_per_warp_m; ++mma_m)
                {
                    asm volatile (
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                        "{%0, %1}, "
                        "{%2, %3}, "
                        "{%4}, "
                        "{%5, %6};"
                        : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
                        : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
                        "r"(B_register[mma_k][mma_n]),
                        "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
                    );
                }
            }
        }

        consumer_block = consumer_block < STAGES - 1 ? consumer_block + 1 : 0;
    }

    //////////////
    // epilogue //
    //////////////

    half alpha_ = (half)alpha;
    half beta_ = (half)beta;

    half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];

    half* C_block_gmem = C + block_m * BM * CD_stride + block_n * BN;
    half* C_warp_gmem = C_block_gmem + warp_m * WM * CD_stride + warp_n * WN;

    half* D_block_gmem = D + block_m * BM * CD_stride + block_n * BN;
    half* D_warp_gmem = D_block_gmem + warp_m * WM * CD_stride + warp_n * WN;

    for (uint32_t mma_m = 0; mma_m < mma_tiles_per_warp_m; ++mma_m)
    {
        for (uint32_t mma_n = 0; mma_n < mma_tiles_per_warp_n; ++mma_n)
        {
            half* C_mma_gmem = C_warp_gmem + mma_m * MMA_M * CD_stride + mma_n * MMA_N;
            ldmatrix_m16n8_gmem(C_mma_gmem, C_register[mma_m][mma_n], CD_stride * sizeof(half));

            acc_register_[mma_m][mma_n][0] = alpha_ * acc_register_[mma_m][mma_n][0] + beta_ * C_register[mma_m][mma_n][
                0];
            acc_register_[mma_m][mma_n][1] = alpha_ * acc_register_[mma_m][mma_n][1] + beta_ * C_register[mma_m][mma_n][
                1];
            acc_register_[mma_m][mma_n][2] = alpha_ * acc_register_[mma_m][mma_n][2] + beta_ * C_register[mma_m][mma_n][
                2];
            acc_register_[mma_m][mma_n][3] = alpha_ * acc_register_[mma_m][mma_n][3] + beta_ * C_register[mma_m][mma_n][
                3];
        }
    }

    for (uint32_t mma_m = 0; mma_m < mma_tiles_per_warp_m; ++mma_m)
    {
        for (uint32_t mma_n = 0; mma_n < mma_tiles_per_warp_n; ++mma_n)
        {
            half* D_mma_gmem = D_warp_gmem + mma_m * MMA_M * CD_stride + mma_n * MMA_N;
            stmatrix_m16n8(D_mma_gmem, acc_register_[mma_m][mma_n], CD_stride * sizeof(half));
        }
    }
}

void launch_matrix_multiplication(const uint32_t M, const uint32_t N, const uint32_t K,
                                  const float alpha, half* A, half* B, const float beta, half* C, half* D,
                                  cudaStream_t stream)
{
    constexpr uint32_t BM = 256;
    constexpr uint32_t BN = 256;
    constexpr uint32_t BK = 32;

    constexpr uint32_t MMA_M = 16;
    constexpr uint32_t MMA_N = 8;
    constexpr uint32_t MMA_K = 8;

    constexpr uint32_t WARPS_PER_BLOCK_M = 2;
    constexpr uint32_t WARPS_PER_BLOCK_N = 4;
    constexpr uint32_t WARPS_PER_BLOCK_K = 1;

    constexpr uint32_t WM = BM / WARPS_PER_BLOCK_M;
    constexpr uint32_t WN = BN / WARPS_PER_BLOCK_N;
    constexpr uint32_t WK = BK / WARPS_PER_BLOCK_K;

    const uint32_t BLOCK_M = M / BM;
    const uint32_t BLOCK_N = N / BN;

    constexpr uint32_t WARP_SIZE = 32;

    constexpr uint32_t THREAD_M = WARPS_PER_BLOCK_M;
    constexpr uint32_t THREAD_N = WARPS_PER_BLOCK_N * WARP_SIZE;
    constexpr uint32_t NUM_THREADS = THREAD_M * THREAD_N;

    constexpr uint32_t STAGES = 2;

    constexpr uint32_t smem_size = STAGES * (BM * BK + BN * BK) * sizeof(half);

    dim3 grid_dim(BLOCK_N, BLOCK_M);
    dim3 block_dim(THREAD_N, THREAD_M);

    CHECK_CUDA_ERROR(
        cudaFuncSetAttribute(matrix_multiplication<BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NUM_THREADS, STAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    matrix_multiplication<BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NUM_THREADS, STAGES>
        <<<grid_dim, block_dim, smem_size, stream>>>(M, N, K, alpha, A, B, beta, C, D);
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    uint32_t M = FLAGS_M;
    uint32_t N = FLAGS_N;
    uint32_t K = FLAGS_K;
    auto alpha = static_cast<float>(FLAGS_alpha);
    auto beta = static_cast<float>(FLAGS_beta);

    bool test = FLAGS_test;
    uint32_t num_warmups = FLAGS_num_warmups;
    uint32_t num_repeats = FLAGS_num_repeats;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    auto [host_params, device_params] = setup_params<half>(M, N, K, alpha, beta);

    std::function<void(cudaStream_t)> bound_function_matrix_multiplication{
        std::bind(launch_matrix_multiplication, device_params.M, device_params.N, device_params.K,
                  device_params.alpha, device_params.A, device_params.B,
                  device_params.beta, device_params.C, device_params.D, stream)
    };

    float const latency_gpu{
        measure_performance(bound_function_matrix_multiplication, stream, num_repeats, num_warmups)
    };

    std::cout << "Latency for matrix multiplication on GPU: " << latency_gpu << std::endl;

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    if (test)
    {
        std::clock_t time_start = std::clock();
        matrix_multiplication_cpu(host_params.M, host_params.N, host_params.K,
                                  host_params.alpha, host_params.A, host_params.B,
                                  host_params.beta, host_params.C, host_params.D);
        std::clock_t time_end = std::clock();

        double latency_cpu = (double)(time_end - time_start) / CLOCKS_PER_SEC * 1000;
        std::cout << "Latency for matrix multiplication on CPU: " << latency_cpu << std::endl;

        if (check_result(device_params.D, host_params.D, M, N, 1e-4))
        {
            std::cout << "Result is correct" << std::endl;
        }
        else
        {
            std::cout << "Result is incorrect" << std::endl;
        }
    }

    free_params(host_params, device_params);
    return 0;
}
