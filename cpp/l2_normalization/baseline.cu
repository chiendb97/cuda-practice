#include <cub/block/block_reduce.cuh>
#include "../cuda_common.cuh"

__forceinline__ __device__ void multiply_accumulate(const float4 &a, const float4 &b, float &sum) {
    sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<int block_size>
__global__
void l2_norm(const float4 *__restrict__ X, float4 *Y, size_t B, size_t N, float eps) {
    auto ti = blockIdx.x;
    auto di = threadIdx.x;

    if (ti >= B) return;

    X += ti * N;
    float sum = 0;
    float4 vec;
    for (auto i = di; i < N; i += block_size) {
        vec = X[i];
        multiply_accumulate(vec, vec, sum);
    }

    using BlockReduce = cub::BlockReduce<float, block_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) shared_inv_rms = rsqrtf(sum + eps);
    __syncthreads();

    float inv_rms = shared_inv_rms;
    Y += ti * N;
    for (auto i = di; i < N; i += block_size) {
        vec = X[i];
        vec.x *= inv_rms; vec.y *= inv_rms; vec.z *= inv_rms; vec.w *= inv_rms;
        Y[i] = vec;
    }
}

template __global__ void l2_norm<128>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l2_norm<256>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l2_norm<512>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l2_norm<1024>(const float4 *__restrict__, float4 *, size_t, size_t, float);

void launch_l2_norm(const float *d_X, float *d_output, size_t b, size_t n, float eps,
                    int grid_dim, int block_dim, cudaStream_t stream) {
    auto X4 = reinterpret_cast<const float4 *>(d_X);
    auto Y4 = reinterpret_cast<float4 *>(d_output);
    if (block_dim == 128)       l2_norm<128><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n / 4, eps);
    else if (block_dim == 256)  l2_norm<256><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n / 4, eps);
    else if (block_dim == 512)  l2_norm<512><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n / 4, eps);
    else if (block_dim == 1024) l2_norm<1024><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n / 4, eps);
    else std::cerr << "Unsupported block_dim: " << block_dim << "\n";
}
