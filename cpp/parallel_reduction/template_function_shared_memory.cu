#include "../cuda_common.cuh"

template<unsigned int BlockDim>
__global__
void reduce_template_shared_memory(int *a, int *s, int size) {
    __shared__ int idata[BlockDim];
    auto gid = blockDim.x * blockIdx.x * 8 + threadIdx.x;

    if (gid + 7 * blockDim.x < size) {
        a[gid] = a[gid] + a[gid + blockDim.x] + a[gid + 2*blockDim.x] + a[gid + 3*blockDim.x]
               + a[gid + 4*blockDim.x] + a[gid + 5*blockDim.x] + a[gid + 6*blockDim.x] + a[gid + 7*blockDim.x];
    }

    idata[threadIdx.x] = a[blockDim.x * blockIdx.x * 8 + threadIdx.x];
    __syncthreads();

    if (BlockDim >= 1024 && threadIdx.x < 512) idata[threadIdx.x] += idata[threadIdx.x + 512];
    __syncthreads();
    if (BlockDim >= 512  && threadIdx.x < 256) idata[threadIdx.x] += idata[threadIdx.x + 256];
    __syncthreads();
    if (BlockDim >= 256  && threadIdx.x < 128) idata[threadIdx.x] += idata[threadIdx.x + 128];
    __syncthreads();
    if (BlockDim >= 128  && threadIdx.x < 64)  idata[threadIdx.x] += idata[threadIdx.x + 64];
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

void launch_reduce_template_function_shared_memory(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    switch (block_dim) {
        case 1024: reduce_template_shared_memory<1024><<<grid_dim, block_dim, 0, stream>>>(a, s, size); break;
        case 512:  reduce_template_shared_memory<512><<<grid_dim, block_dim, 0, stream>>>(a, s, size);  break;
        case 256:  reduce_template_shared_memory<256><<<grid_dim, block_dim, 0, stream>>>(a, s, size);  break;
        case 128:  reduce_template_shared_memory<128><<<grid_dim, block_dim, 0, stream>>>(a, s, size);  break;
        case 64:   reduce_template_shared_memory<64><<<grid_dim, block_dim, 0, stream>>>(a, s, size);   break;
        case 32:   reduce_template_shared_memory<32><<<grid_dim, block_dim, 0, stream>>>(a, s, size);   break;
        default: break;
    }
}
