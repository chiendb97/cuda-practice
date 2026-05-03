#include "../cuda_common.cuh"

__inline__ __device__ int warp_reduce(int sum) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    return sum;
}

__global__
void reduce_warp_shuffle_instruction(int *a, int *s, int size) {
    extern __shared__ int smem[];
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= size) {
        return;
    }

    auto sum = a[gid];

    auto land_idx = threadIdx.x % 32;
    auto warp_idx = threadIdx.x / 32;

    sum = warp_reduce(sum);

    if (land_idx == 0) {
        smem[warp_idx] = sum;
    }

    __syncthreads();

    sum = threadIdx.x < 32 ? smem[threadIdx.x] : 0;

    if (warp_idx == 0) {
        sum = warp_reduce(sum);
    }

    if (threadIdx.x == 0) {
        s[blockIdx.x] = sum;
    }
}

void launch_reduce_warp_shuffle_instruction(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    reduce_warp_shuffle_instruction<<<grid_dim, block_dim, block_dim / 32 * sizeof(int), stream>>>(a, s, size);
}
