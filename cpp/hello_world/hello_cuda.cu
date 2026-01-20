//
// Created by chiendb on 3/4/24.
//

#include <iostream>
#include <cstdio>
#include <gflags/gflags.h>

DEFINE_int32(nx, 128, "Number of blocks in x direction");
DEFINE_int32(ny, 64, "Number of blocks in y direction");
DEFINE_int32(block_x, 8, "Number of threads in x direction");
DEFINE_int32(block_y, 4, "Number of threads in y direction");

__global__ void hello_cuda() {
    printf("Hello CUDA world, %d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}


int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    int nx = FLAGS_nx;
    int ny = FLAGS_ny;

    dim3 block(FLAGS_block_x, FLAGS_block_y);
    dim3 grid(FLAGS_nx / FLAGS_block_x, FLAGS_ny / FLAGS_block_y);

    hello_cuda<<<grid, block>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}