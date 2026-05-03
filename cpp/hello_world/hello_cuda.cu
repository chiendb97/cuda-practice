#include "../cuda_common.cuh"

__global__ void hello_cuda() {
    printf("Hello CUDA world, %d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

void launch_hello_cuda(int nx, int ny, int block_x, int block_y, cudaStream_t stream) {
    dim3 block(block_x, block_y);
    dim3 grid(nx / block_x, ny / block_y);
    hello_cuda<<<grid, block, 0, stream>>>();
}
