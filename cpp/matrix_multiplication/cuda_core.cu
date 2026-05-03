#include "../cuda_common.cuh"

template <typename T>
__global__ void matrix_multiplication_kernel(int M, int N, int K,
                                             const T *A, const T *B, T *C) {
    auto m = blockIdx.x * blockDim.x + threadIdx.x;
    auto n = blockIdx.y * blockDim.y + threadIdx.y;

    T accum = 0;
    for (unsigned k = 0; k < K; ++k)
        accum += A[m + k * M] * B[n + k * N];

    C[m + n * M] = accum;
}

template <typename T, int block_size>
__global__ void matrix_multiplication_optimized_kernel(int M, int N, int K,
                                                       const T *A, const T *B, T *C) {
    __shared__ T tile_a[block_size][block_size + 1];
    __shared__ T tile_b[block_size][block_size + 1];

    auto m = blockIdx.x * blockDim.x + threadIdx.x;
    auto n = blockIdx.y * blockDim.y + threadIdx.y;

    T accum = 0;
    for (unsigned block = 0; block < K; block += block_size) {
        tile_a[threadIdx.x][threadIdx.y] = A[m + (block + threadIdx.y) * M];
        tile_b[threadIdx.y][threadIdx.x] = B[n + (block + threadIdx.x) * N];
        __syncthreads();
        for (unsigned i = 0; i < block_size; ++i)
            accum += tile_a[threadIdx.x][i] * tile_b[threadIdx.y][i];
        __syncthreads();
    }

    C[m + n * M] = accum;
}

void launch_matmul_cuda_core(int M, int N, int K,
                             const float *A, const float *B, float *C,
                             cudaStream_t stream) {
    constexpr int block_size = 32;
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((M - 1) / block_size + 1, (N - 1) / block_size + 1);
    matrix_multiplication_kernel<<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, B, C);
}

void launch_matmul_cuda_core_opt(int M, int N, int K,
                                 const float *A, const float *B, float *C,
                                 cudaStream_t stream) {
    constexpr int block_size = 32;
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((M - 1) / block_size + 1, (N - 1) / block_size + 1);
    matrix_multiplication_optimized_kernel<float, block_size><<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, B, C);
}
