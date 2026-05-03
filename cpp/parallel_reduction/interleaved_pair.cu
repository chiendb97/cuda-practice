#include "../cuda_common.cuh"

__global__
void reduce_interleaved_pair(int *a, int *s, int size) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= size) return;
    auto *idata = a + blockDim.x * blockIdx.x;
    for (auto offset = blockDim.x / 2; offset >= 1; offset >>= 1) {
        if (threadIdx.x < offset)
            idata[threadIdx.x] += idata[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) s[blockIdx.x] = idata[0];
}

void launch_reduce_interleaved_pair(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    reduce_interleaved_pair<<<grid_dim, block_dim, 0, stream>>>(a, s, size);
}
