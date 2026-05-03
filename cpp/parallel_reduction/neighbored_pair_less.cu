#include "../cuda_common.cuh"

__global__
void reduce_neighbored_pair_less(int *a, int *s, int size) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= size) return;
    auto *idata = a + blockDim.x * blockIdx.x;
    for (auto offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = 2 * threadIdx.x * offset;
        if (idx < blockDim.x)
            idata[idx] += idata[idx + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) s[blockIdx.x] = idata[0];
}

void launch_reduce_neighbored_pair_less(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    reduce_neighbored_pair_less<<<grid_dim, block_dim, 0, stream>>>(a, s, size);
}
