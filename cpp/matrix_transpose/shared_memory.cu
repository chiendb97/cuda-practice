#include "../cuda_common.cuh"

__global__
void transpose_shared_memory(int *source, int *target, int M, int N) {
    __shared__ int tile[16][16];
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = source[iy * M + ix];
    __syncthreads();
    ix = blockDim.y * blockIdx.y + threadIdx.x;
    iy = blockDim.x * blockIdx.x + threadIdx.y;
    if (ix < M && iy < N) target[iy * N + ix] = tile[threadIdx.x][threadIdx.y];
}

__global__
void transpose_shared_memory_padding(int *source, int *target, int M, int N) {
    __shared__ int tile[16][17];
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = source[iy * M + ix];
    __syncthreads();
    ix = blockDim.y * blockIdx.y + threadIdx.x;
    iy = blockDim.x * blockIdx.x + threadIdx.y;
    if (ix < M && iy < N) target[iy * N + ix] = tile[threadIdx.x][threadIdx.y];
}

void launch_transpose_shared_memory(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_shared_memory<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
void launch_transpose_shared_memory_padding(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_shared_memory_padding<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
