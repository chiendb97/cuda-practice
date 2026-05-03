#include "../cuda_common.cuh"

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 4
#endif

template<typename T, int N>
struct Array {
    T __a[N];
    __host__ __device__ constexpr T& operator[](int i) noexcept { return __a[i]; }
    __host__ __device__ constexpr const T& operator[](int i) const noexcept { return __a[i]; }
};

template<typename T, int N>
inline __device__ void Load(Array<T, N> &dst, const T *src) {
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4))
        (uint4 &) dst = *(const uint4 *) src;
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2))
        (uint2 &) dst = *(const uint2 *) src;
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint))
        (uint1 &) dst = *(const uint1 *) src;
    else {
        constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
        for (int i = 0; i < M; ++i)
            *((uint4 *) &dst + i) = *((const uint4 *) src + i);
    }
}

template<typename T, int N>
inline __device__ void Store(T *dst, const Array<T, N> &src) {
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4))
        *(uint4 *) dst = (const uint4 &) src;
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2))
        *(uint2 *) dst = (const uint2 &) src;
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint1))
        *(uint1 *) dst = (const uint1 &) src;
    else {
        constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
        for (int i = 0; i < M; ++i)
            *((uint4 *) dst + i) = *((const uint4 *) &src + i);
    }
}

template<typename T>
__device__ __forceinline__ T warp_prefix_sum(T val, int lane_id) {
    T x = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        T y = __shfl_up_sync(0xffffffff, x, offset);
        if (lane_id >= offset) x += y;
    }
    return x - val;
}

template<typename T, int block_dim>
__device__ T block_prefix_sum(T val) {
    int lane_id = threadIdx.x & 0x1f;
    int warp_id = threadIdx.x >> 5;
    __shared__ T shared[block_dim >> 5];
    T warp_prefix = warp_prefix_sum(val, lane_id);
    if (lane_id == 31) shared[warp_id] = warp_prefix + val;
    __syncthreads();
    if (threadIdx.x < 32) shared[threadIdx.x] = warp_prefix_sum(shared[threadIdx.x], lane_id);
    __syncthreads();
    return warp_prefix + shared[warp_id];
}

template<typename T, int block_dim, int items_per_thread>
__global__
void prefix_sum_opt(const T *__restrict__ X, T *Y, int *flags, T *scan_value, int *block_counter, size_t N) {
    __shared__ size_t sbid;
    if (threadIdx.x == 0) sbid = atomicAdd(block_counter, 1);
    __syncthreads();
    size_t bid = sbid;
    auto idx = bid * blockDim.x + threadIdx.x;

    Array<T, items_per_thread> vec{};
    Load(vec, X + idx * items_per_thread);
#pragma unroll
    for (int i = 1; i < items_per_thread; ++i) vec[i] += vec[i - 1];

    auto reduced_total_sum = block_prefix_sum<T, block_dim>(vec[items_per_thread - 1]);

    __shared__ T prev_sum;
    if (threadIdx.x == block_dim - 1) {
        while (bid != 0 && atomicAdd(&flags[bid], 0) == 0) {}
        prev_sum = scan_value[bid];
        scan_value[bid + 1] = prev_sum + reduced_total_sum + vec[items_per_thread - 1];
        __threadfence();
        atomicAdd(&flags[bid + 1], 1);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < items_per_thread; ++i) vec[i] = reduced_total_sum + prev_sum + vec[i];
    Store(Y + idx * items_per_thread, vec);
}

template __global__ void prefix_sum_opt<float, 128, ITEMS_PER_THREAD>(const float *, float *, int *, float *, int *, size_t);
template __global__ void prefix_sum_opt<float, 256, ITEMS_PER_THREAD>(const float *, float *, int *, float *, int *, size_t);
template __global__ void prefix_sum_opt<float, 512, ITEMS_PER_THREAD>(const float *, float *, int *, float *, int *, size_t);
template __global__ void prefix_sum_opt<float, 1024, ITEMS_PER_THREAD>(const float *, float *, int *, float *, int *, size_t);

void launch_prefix_sum_opt(const float *d_X, float *d_output, int n, int *flags, float *scan_value, int *block_counter,
                           int grid_dim, int block_dim, cudaStream_t stream) {
    if (block_dim == 128)
        prefix_sum_opt<float, 128, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(d_X, d_output, flags, scan_value, block_counter, n);
    else if (block_dim == 256)
        prefix_sum_opt<float, 256, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(d_X, d_output, flags, scan_value, block_counter, n);
    else if (block_dim == 512)
        prefix_sum_opt<float, 512, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(d_X, d_output, flags, scan_value, block_counter, n);
    else if (block_dim == 1024)
        prefix_sum_opt<float, 1024, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(d_X, d_output, flags, scan_value, block_counter, n);
}
