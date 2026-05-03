#include "../cuda_common.cuh"

__global__
void relu_baseline(const float4 *__restrict__ X, float4 *Y, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 vec = X[idx];
    vec.x = fmaxf(vec.x, 0.0f);
    vec.y = fmaxf(vec.y, 0.0f);
    vec.z = fmaxf(vec.z, 0.0f);
    vec.w = fmaxf(vec.w, 0.0f);
    Y[idx] = vec;
}

void launch_relu(const float *d_X, float *d_output, size_t n, size_t m,
                 int grid_dim, int block_dim, cudaStream_t stream) {
    relu_baseline<<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const float4 *>(d_X),
        reinterpret_cast<float4 *>(d_output), n * m);
}
