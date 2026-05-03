#include "../cuda_common.cuh"

__global__
void copy_row(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) target[iy * M + ix] = source[iy * M + ix];
}

__global__
void copy_column(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) target[ix * N + iy] = source[ix * N + iy];
}

__global__
void transpose_row(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) target[ix * N + iy] = source[iy * M + ix];
}

__global__
void transpose_column(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) target[iy * M + ix] = source[ix * N + iy];
}

__global__
void transpose_diagonal_row(int *source, int *target, int M, int N) {
    auto block_idx_y = blockIdx.x;
    auto block_idx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    auto ix = blockDim.x * block_idx_x + threadIdx.x;
    auto iy = blockDim.y * block_idx_y + threadIdx.y;
    if (ix < M && iy < N) target[ix * N + iy] = source[iy * M + ix];
}

__global__
void transpose_diagonal_column(int *source, int *target, int M, int N) {
    auto block_idx_y = blockIdx.x;
    auto block_idx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    auto ix = blockDim.x * block_idx_x + threadIdx.x;
    auto iy = blockDim.y * block_idx_y + threadIdx.y;
    if (ix < M && iy < N) target[iy * M + ix] = source[ix * N + iy];
}

void launch_copy_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    copy_row<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
void launch_copy_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    copy_column<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
void launch_transpose_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_row<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
void launch_transpose_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_column<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
void launch_transpose_diagonal_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_diagonal_row<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
void launch_transpose_diagonal_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_diagonal_column<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}
