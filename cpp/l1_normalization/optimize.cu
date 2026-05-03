#include "../cuda_common.cuh"

__forceinline__ __device__ void accumulate_abs_opt(const float4 &a, float &sum) {
    sum += fabsf(a.x) + fabsf(a.y) + fabsf(a.z) + fabsf(a.w);
}

__forceinline__ __device__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<int block_size>
__launch_bounds__(block_size, 2)
__global__ void l1_norm_opt(const float4 *__restrict__ X, float4 *Y, size_t B, size_t N, float eps) {
    constexpr int warp_size = 32;
    const int lane    = threadIdx.x & (warp_size - 1);
    const int warp_id = threadIdx.x / warp_size;
    const int num_warps = block_size / warp_size;

    const auto ti = blockIdx.x;
    if (ti >= B) return;

    X += ti * N;
    float sum = 0.f;
    for (size_t i = threadIdx.x; i < N; i += block_size) {
        float4 vec = X[i];
        accumulate_abs_opt(vec, sum);
    }

    sum = warp_sum(sum);

    __shared__ float warp_partials[block_size / warp_size];
    if (lane == 0) warp_partials[warp_id] = sum;
    __syncthreads();

    float block_sum = 0.f;
    if (warp_id == 0) {
        block_sum = (lane < num_warps) ? warp_partials[lane] : 0.f;
        block_sum = warp_sum(block_sum);
    }

    float inv_norm = 0.f;
    if (warp_id == 0 && lane == 0) {
        inv_norm = 1.0f / (block_sum + eps);
        warp_partials[0] = inv_norm;
    }
    __syncthreads();
    inv_norm = warp_partials[0];

    Y += ti * N;
    for (size_t i = threadIdx.x; i < N; i += block_size) {
        float4 vec = X[i];
        vec.x *= inv_norm; vec.y *= inv_norm; vec.z *= inv_norm; vec.w *= inv_norm;
        Y[i] = vec;
    }
}

template __global__ void l1_norm_opt<128>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l1_norm_opt<256>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l1_norm_opt<512>(const float4 *__restrict__, float4 *, size_t, size_t, float);
template __global__ void l1_norm_opt<1024>(const float4 *__restrict__, float4 *, size_t, size_t, float);

void launch_l1_norm_opt(const float *d_X, float *d_output, size_t b, size_t n, float eps,
                        int grid_dim, int block_dim, cudaStream_t stream) {
    auto X4 = reinterpret_cast<const float4 *>(d_X);
    auto Y4 = reinterpret_cast<float4 *>(d_output);
    size_t n4 = n / 4;
    if (block_dim == 128)       l1_norm_opt<128><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else if (block_dim == 256)  l1_norm_opt<256><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else if (block_dim == 512)  l1_norm_opt<512><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else if (block_dim == 1024) l1_norm_opt<1024><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    else printf("Unsupported block_dim: %d\n", block_dim);
}
