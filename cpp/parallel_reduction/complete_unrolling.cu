#include "../cuda_common.cuh"

__global__
void reduce_complete_unrolling(int *a, int *s, int size) {
    auto gid   = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    auto *idata = a + blockDim.x * blockIdx.x * 8;

    if (gid + 7 * blockDim.x < size) {
        a[gid] = a[gid] + a[gid + blockDim.x] + a[gid + 2*blockDim.x] + a[gid + 3*blockDim.x]
               + a[gid + 4*blockDim.x] + a[gid + 5*blockDim.x] + a[gid + 6*blockDim.x] + a[gid + 7*blockDim.x];
    }
    __syncthreads();

    if (blockDim.x >= 1024 && threadIdx.x < 512) idata[threadIdx.x] += idata[threadIdx.x + 512];
    __syncthreads();
    if (blockDim.x >= 512  && threadIdx.x < 256) idata[threadIdx.x] += idata[threadIdx.x + 256];
    __syncthreads();
    if (blockDim.x >= 256  && threadIdx.x < 128) idata[threadIdx.x] += idata[threadIdx.x + 128];
    __syncthreads();
    if (blockDim.x >= 128  && threadIdx.x < 64)  idata[threadIdx.x] += idata[threadIdx.x + 64];
    __syncthreads();

    if (threadIdx.x < 32) {
        volatile int *vsmem = idata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }
    if (threadIdx.x == 0) s[blockIdx.x] = idata[0];
}

void launch_reduce_complete_unrolling(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    reduce_complete_unrolling<<<grid_dim, block_dim, 0, stream>>>(a, s, size);
}
