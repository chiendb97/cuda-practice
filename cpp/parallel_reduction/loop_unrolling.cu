#include "../cuda_common.cuh"

__global__
void reduce_loop_unrolling(int *a, int *s, int size) {
    auto gid   = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    auto *idata = a + blockDim.x * blockIdx.x * 8;

    if (gid + 7 * blockDim.x < size) {
        a[gid] = a[gid] + a[gid + blockDim.x] + a[gid + 2*blockDim.x] + a[gid + 3*blockDim.x]
               + a[gid + 4*blockDim.x] + a[gid + 5*blockDim.x] + a[gid + 6*blockDim.x] + a[gid + 7*blockDim.x];
    }
    __syncthreads();

    for (auto offset = blockDim.x / 2; offset >= 1; offset >>= 1) {
        if (threadIdx.x < offset)
            idata[threadIdx.x] += idata[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) s[blockIdx.x] = idata[0];
}

void launch_reduce_loop_unrolling(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    reduce_loop_unrolling<<<grid_dim, block_dim, 0, stream>>>(a, s, size);
}
