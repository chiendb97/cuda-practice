#include "../cuda_common.cuh"

template <typename T, int NUM>
__forceinline__ __device__ T warp_reduce_sum(T *val) {
#pragma unroll
    for (int i = 0; i < NUM; ++i) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
    }
    return (T)(0.0f);
}

template <typename T, int NUM>
__forceinline__ __device__ T block_reduce_sum(T *val) {
    __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid  = threadIdx.x >> 5;
    warp_reduce_sum<T, NUM>(val);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; ++i) shared[i][wid] = val[i];
    }
    __syncthreads();
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; ++i)
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    warp_reduce_sum<T, NUM>(val);
    return (T)(0.0f);
}

__global__ void rms_norm_warp(const float4 *__restrict__ X, float4 *Y, size_t B,
                               size_t N, float invN, float eps) {
    auto ti = blockIdx.x;
    auto di = threadIdx.x;

    if (ti >= B) return;

    X += ti * N;
    float sum[1] = {0};
    float4 vec;
    for (auto i = di; i < N; i += blockDim.x) {
        vec = __ldg(&X[i]);
        sum[0] += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
    }

    if (blockDim.x < 32) warp_reduce_sum<float, 1>(sum);
    else                 block_reduce_sum<float, 1>(sum);

    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) shared_inv_rms = rsqrtf(sum[0] * invN + eps);
    __syncthreads();

    float inv_rms = shared_inv_rms;
    Y += ti * N;
    for (auto i = di; i < N; i += blockDim.x) {
        vec = __ldg(&X[i]);
        vec.x *= inv_rms; vec.y *= inv_rms; vec.z *= inv_rms; vec.w *= inv_rms;
        Y[i] = vec;
    }
}

void launch_rms_norm_warp(const float *d_X, float *d_output, size_t b, size_t n,
                          float eps, int grid_dim, int block_dim, cudaStream_t stream) {
    rms_norm_warp<<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const float4 *>(d_X),
        reinterpret_cast<float4 *>(d_output), b, n / 4, 1.0f / n, eps);
}
