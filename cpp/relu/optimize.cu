#include "../cuda_common.cuh"

__global__
void relu_opt(const int4 *__restrict__ X, int4 *Y, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    int4 vec = X[idx];
    vec.x = vec.x & ~(vec.x >> 31);
    vec.y = vec.y & ~(vec.y >> 31);
    vec.z = vec.z & ~(vec.z >> 31);
    vec.w = vec.w & ~(vec.w >> 31);
    Y[idx] = vec;
}

void launch_relu_opt(const float *d_X, float *d_output, size_t n, size_t m,
                     int grid_dim, int block_dim, cudaStream_t stream) {
    relu_opt<<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const int4 *>(d_X),
        reinterpret_cast<int4 *>(d_output), n * m);
}
