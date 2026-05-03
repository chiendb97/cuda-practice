#include <cub/block/block_reduce.cuh>
#include "../cuda_common.cuh"

__forceinline__ __device__ void accumulate_abs(const float4 &a, float &sum) {
    sum += fabsf(a.x) + fabsf(a.y) + fabsf(a.z) + fabsf(a.w);
}

template<int block_size>
__global__ void l1_norm(const float4 *__restrict__ X, float4 *Y, size_t B, size_t N, float eps) {
    const auto ti = blockIdx.x;
    const auto di = threadIdx.x;

    if (ti >= B) return;

    X += ti * N;
    float sum = 0.f;
    for (size_t i = di; i < N; i += block_size) {
        float4 vec = X[i];
        accumulate_abs(vec, sum);
    }

    using BlockReduce = cub::BlockReduce<float, block_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_inv_norm;
    if (threadIdx.x == 0) shared_inv_norm = 1.0f / (sum + eps);
    __syncthreads();

    float inv_norm = shared_inv_norm;
    Y += ti * N;
    for (size_t i = di; i < N; i += block_size) {
        float4 vec = X[i];
        vec.x *= inv_norm; vec.y *= inv_norm; vec.z *= inv_norm; vec.w *= inv_norm;
        Y[i] = vec;
    }
}

template __global__ void l1_norm<128>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l1_norm<256>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l1_norm<512>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l1_norm<1024>(const float4 *__restrict__, float4 *, size_t, size_t, float);

void launch_l1_norm(const float *d_X, float *d_output, size_t b, size_t n, float eps,
                    int grid_dim, int block_dim, cudaStream_t stream) {
    auto X4 = reinterpret_cast<const float4 *>(d_X);
    auto Y4 = reinterpret_cast<float4 *>(d_output);
    size_t n4 = n / 4;
    if (block_dim == 128)       l1_norm<128><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else if (block_dim == 256)  l1_norm<256><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else if (block_dim == 512)  l1_norm<512><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else if (block_dim == 1024) l1_norm<1024><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else printf("Unsupported block_dim: %d\n", block_dim);
}
