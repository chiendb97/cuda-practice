//
// Created by chiendb on 3/4/24.
//

#include <iostream>
#include <cstdio>

__global__ void hello_cuda() {
    printf("Hello CUDA world, %d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}


int main() {
    int nx = 128;
    int ny = 64;

    dim3 block(8, 4);
    dim3 grid(nx / block.x, ny / block.y);
    hello_cuda<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}