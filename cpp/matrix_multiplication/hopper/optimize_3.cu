//
// Created by root on 7/15/25.
//

#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <gflags/gflags.h>

#include "device_utils.cuh"
#include "host_utils.cuh"

DEFINE_uint32(M, 512, "M");
DEFINE_uint32(N, 512, "N");
DEFINE_uint32(K, 512, "K");
DEFINE_double(alpha, 1.0, "alpha");
DEFINE_double(beta, 0.0, "beta");

DEFINE_bool(test, false, "test");
DEFINE_uint32(num_warmups, 0, "num_warmups");
DEFINE_uint32(num_repeats, 1, "num_repeats");

struct BlockCoordinate
{
    uint32_t block_m;
    uint32_t block_n;
};

template <uint32_t BM, uint32_t BN, uint32_t TILE_M, uint32_t TILE_N>
__device__ BlockCoordinate get_block_coordinate(uint32_t iter, uint32_t M, uint32_t N)
{
    uint32_t block_n = blockIdx.x / TILE_M;
    uint32_t block_m = blockIdx.x % TILE_M;

    constexpr uint32_t m_elements_per_tile = TILE_M * BM;
    uint32_t m_iter = M / m_elements_per_tile;

    uint32_t m = block_m + (iter % m_iter) * TILE_M;
    uint32_t n = block_n + (iter / m_iter) * TILE_N;

    assert(n < (N / BN));
    assert(m < (M / BM));

    return BlockCoordinate{m, n};
}

// template <uint32_t BM, uint32_t BN, uint32_t TILE_M, uint32_t TILE_N>
// __device__ BlockCoordinate get_block_coordinate(uint32_t iter, uint32_t M, uint32_t N)
// {
//     uint32_t block_m = blockIdx.x % TILE_N;
//     uint32_t block_n = blockIdx.x / TILE_N;

//     constexpr uint32_t n_elements_per_tile = TILE_N * BN;
//     uint32_t n_iter = N / n_elements_per_tile;

//     uint32_t n = block_m + (iter % n_iter) * TILE_N;
//     uint32_t m = block_n + (iter / n_iter) * TILE_M;

//     assert(n < (N / BN));
//     assert(m < (M / BM));

//     return BlockCoordinate{m, n};
// }

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t BP, uint32_t TILE_M, uint32_t TILE_N, uint32_t NUM_CONSUMER_WARP_GROUPS>
__global__ void
matrix_multiplication(const uint32_t M, const uint32_t N, const uint32_t K,
                      const float alpha, half *A, half *B, const float beta, half *C, half *D,
                      CUtensorMap *A_tensor_map, CUtensorMap *B_tensor_map)
{
    constexpr uint32_t WGMMA_M = 64;
    constexpr uint32_t WGMMA_N = 256;
    constexpr uint32_t WGMMA_K = 16;

    const uint32_t warp_group_idx = threadIdx.x / 128;

    bool is_producer = (threadIdx.x < 128);

    // __shared__ alignas(128) half A_block_smem[BM * BK * BP];
    // __shared__ alignas(128) half B_block_smem[BN * BK * BP];

    extern __shared__ __align__(128) half smem[];
    half *A_block_smem = smem;
    half *B_block_smem = smem + BM * BK * BP;

    constexpr uint32_t pipeline_stage_elements = BM * BK + BN * BK;
    constexpr uint32_t pipeline_stage_num_bytes = pipeline_stage_elements * sizeof(half);

    __shared__ cuda::barrier<cuda::thread_scope_block> empty_barrier[BP];
    __shared__ cuda::barrier<cuda::thread_scope_block> full_barrier[BP];

    if (threadIdx.x == 0)
    {
        for (uint32_t p = 0; p < BP; ++p)
        {
            init(&empty_barrier[p], 1 + NUM_CONSUMER_WARP_GROUPS * 128);
            init(&full_barrier[p], 1 + NUM_CONSUMER_WARP_GROUPS * 128);
        }

        cuda::device::experimental::fence_proxy_async_shared_cta();
    }

    __syncthreads();

    const uint32_t num_iterations = ((M / BM) * (N / BN)) / gridDim.x;

    if (is_producer)
    {
        // set_max_reg_dec<24>();
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(24));

        if (threadIdx.x == 0)
        {
            uint32_t p = 0;
            for (uint32_t iter = 0; iter < num_iterations; ++iter)
            {
                BlockCoordinate block_coordinate = get_block_coordinate<BM, BN, TILE_M, TILE_N>(iter, M, N);
                for (uint32_t block_k = 0; block_k < K / BK; ++block_k)
                {
                    empty_barrier[p].wait(empty_barrier[p].arrive());
                    cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&A_block_smem[p * BM * BK], A_tensor_map, block_k * BK, block_coordinate.block_m * BM, full_barrier[p]);
                    cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&B_block_smem[p * BN * BK], B_tensor_map, block_k * BK, block_coordinate.block_n * BN, full_barrier[p]);

                    cuda::barrier<cuda::thread_scope_block>::arrival_token token = cuda::device::barrier_arrive_tx(full_barrier[p], 1, pipeline_stage_num_bytes);
                    p = (p + 1) % BP;
                }
            }
        }
    }
    else
    {
        // set_max_reg_inc<240>();
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(240));

        float accum_reg[128];
        memset(accum_reg, 0, sizeof(accum_reg));

        const uint32_t consumer_idx = threadIdx.x / 128 - 1;

        for (uint32_t p = 0; p < BP; ++p)
        {
            cuda::barrier<cuda::thread_scope_block>::arrival_token token = empty_barrier[p].arrive();
        }

        uint32_t p = 0;

        for (uint32_t iter = 0; iter < num_iterations; ++iter)
        {
            BlockCoordinate block_coordinate = get_block_coordinate<BM, BN, TILE_M, TILE_N>(iter, M, N);

            for (uint32_t block_k = 0; block_k < K / BK; ++block_k)
            {
                full_barrier[p].wait(full_barrier[p].arrive());

                warp_group_arrive();

                half *A_pipeline_stage_block_smem = A_block_smem + p * BM * BK;
                half *B_pipeline_stage_block_smem = B_block_smem + p * BN * BK;

                half *A_warp_group_smem = A_pipeline_stage_block_smem + consumer_idx * WGMMA_M * BK;

#pragma unroll
                for (uint32_t i = 0; i < BK / WGMMA_K; ++i)
                {
                    uint32_t offset = i * WGMMA_K;
                    wgmma_m64n256k16_f32_f16_f16<BK * sizeof(half)>(accum_reg, A_warp_group_smem + offset, B_pipeline_stage_block_smem + offset);
                }

                wgmma_commit_group();
                wgmma_wait_group<0>();

                cuda::barrier<cuda::thread_scope_block>::arrival_token token = empty_barrier[p].arrive();

                p = (p + 1) % BP;
            }

            half *C_block = C + (block_coordinate.block_m * BM + consumer_idx * WGMMA_M) * N + block_coordinate.block_n * BN;

            uint32_t thread = threadIdx.x % 128;
            uint32_t warp = thread / 32;
            uint32_t lane = thread % 32;

            uint32_t row = (warp * 16) + (lane / 4);
            uint32_t col = (thread % 4) * 2;

#define OUT_IDX(i, j) (i) * N + (j)

            for (uint32_t column_group = 0; column_group < WGMMA_N / 16; ++column_group)
            {
                C_block[OUT_IDX(row, col)] = __float2half(accum_reg[column_group * 8]);
                C_block[OUT_IDX(row, col + 1)] = __float2half(accum_reg[column_group * 8 + 1]);
                C_block[OUT_IDX(row + 8, col)] = __float2half(accum_reg[column_group * 8 + 2]);
                C_block[OUT_IDX(row + 8, col + 1)] = __float2half(accum_reg[column_group * 8 + 3]);
                C_block[OUT_IDX(row, col + 8)] = __float2half(accum_reg[column_group * 8 + 4]);
                C_block[OUT_IDX(row, col + 9)] = __float2half(accum_reg[column_group * 8 + 5]);
                C_block[OUT_IDX(row + 8, col + 8)] = __float2half(accum_reg[column_group * 8 + 6]);
                C_block[OUT_IDX(row + 8, col + 9)] = __float2half(accum_reg[column_group * 8 + 7]);
                col += 16;
            }
#undef OUT_IDX
        }
    }
}

void launch_matrix_multiplication(const uint32_t M, const uint32_t N, const uint32_t K,
                                  const float alpha, half *A, half *B, const float beta, half *C, half *D,
                                  CUtensorMap *A_tensor_map_device, CUtensorMap *B_tensor_map_device,
                                  cudaStream_t stream)
{
    constexpr uint32_t num_consumner_warp_groups = 2;
    constexpr uint32_t BM = 64 * num_consumner_warp_groups;
    constexpr uint32_t BN = 256;
    constexpr uint32_t BK = 64;
    constexpr uint32_t BP = 4;

    constexpr uint32_t TILE_M = 16;
    constexpr uint32_t TILE_N = 8;

    assert(M >= BM * TILE_M);
    assert(N >= BN * TILE_N);

    create_tensor_map<BM, BK>(A, M, K, A_tensor_map_device);
    create_tensor_map<BN, BK>(B, N, K, B_tensor_map_device);

    constexpr uint32_t NUM_THREADS_PER_WARP_GROUP = 128;

    dim3 grid_dim(TILE_M * TILE_N);
    dim3 block_dim(NUM_THREADS_PER_WARP_GROUP * (num_consumner_warp_groups + 1));

    constexpr uint32_t smem_size = BP * (BM * BK + BN * BK) * sizeof(half);

    static_assert(smem_size <= 227000, "sm90 has a max of 227000 bytes of dynamic shared memory");

    CHECK_CUDA_ERROR(cudaFuncSetAttribute(matrix_multiplication<BM, BN, BK, BP, TILE_M, TILE_N, num_consumner_warp_groups>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    matrix_multiplication<BM, BN, BK, BP, TILE_M, TILE_N, num_consumner_warp_groups><<<grid_dim, block_dim, smem_size, stream>>>(M, N, K, alpha, A, B, beta, C, D, A_tensor_map_device, B_tensor_map_device);
}

int main(int argc, char *argv[])
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

    CUtensorMap *A_tensor_map_device = nullptr;
    CUtensorMap *B_tensor_map_device = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&A_tensor_map_device, sizeof(CUtensorMap)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_tensor_map_device, sizeof(CUtensorMap)));

    std::function<void(cudaStream_t)> bound_function_matrix_multiplication{
        std::bind(launch_matrix_multiplication, device_params.M, device_params.N, device_params.K,
                  device_params.alpha, device_params.A, device_params.B,
                  device_params.beta, device_params.C, device_params.D,
                  A_tensor_map_device, B_tensor_map_device, stream)};

    float const latency_gpu{
        measure_performance(bound_function_matrix_multiplication, stream, num_repeats, num_warmups)};

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

        if (check_result(device_params.C, host_params.D, M, N, 1e-4))
        {
            std::cout << "Result is correct" << std::endl;
        }
        else
        {
            std::cout << "Result is incorrect" << std::endl;
        }
    }

    free_params(host_params, device_params);
    CHECK_CUDA_ERROR(cudaFree(A_tensor_map_device));
    CHECK_CUDA_ERROR(cudaFree(B_tensor_map_device));
    return 0;
}
