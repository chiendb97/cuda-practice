//
// Created by chiendb on 3/4/24.
//

#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <gflags/gflags.h>

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 4
#endif

DEFINE_uint32(n, 1 << 20, "n");
DEFINE_uint32(block_dim, 256, "block dim");
DEFINE_uint32(num_warmups, 1, "num_warmups");
DEFINE_uint32(num_repeats, 1, "num_repeats");


#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

void check_cuda_error(cudaError_t err, const char *const func, const char *const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "Cuda Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

void check_last_cuda_error(const char *const file, const int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100) {
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0}; i < num_warmups; ++i) {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0}; i < num_repeats; ++i) {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

void prefix_sum_cpu(const float *X, float *Y, size_t n) {
    Y[0] = X[0];
    for (int i = 1; i < n; ++i) {
        Y[i] = X[i] + Y[i - 1];
    }
}

template<typename T, int N>
struct Array {
    using value_type      = T;
    using size_type       = int;
    using difference_type = int;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    static_assert(N > 0);

    T __a[N];

    __host__ __device__ constexpr reference operator[](size_type i) noexcept
    {
        return __a[i];
    }

    __host__ __device__ constexpr const_reference operator[](size_type i) const noexcept
    {
        return __a[i];
    }

    __host__ __device__ constexpr reference front() noexcept
    {
        return *begin();
    }

    __host__ __device__ constexpr const_reference front() const noexcept
    {
        return *begin();
    }

    __host__ __device__ constexpr reference back() noexcept
    {
        return *(end() - 1);
    }

    __host__ __device__ constexpr const_reference back() const noexcept
    {
        return *(end() - 1);
    }

    __host__ __device__ constexpr pointer data() noexcept
    {
        return &__a[0];
    }

    __host__ __device__ constexpr const_pointer data() const noexcept
    {
        return &__a[0];
    }

    __host__ __device__ constexpr iterator begin() noexcept
    {
        return data();
    }

    __host__ __device__ constexpr const_iterator begin() const noexcept
    {
        return data();
    }

    __host__ __device__ constexpr iterator end() noexcept
    {
        return data() + N;
    }

    __host__ __device__ constexpr const_iterator end() const noexcept
    {
        return data() + N;
    }

    __host__ __device__ static constexpr std::integral_constant<int, N> size() noexcept
    {
        return {};
    }

    __host__ __device__ static constexpr std::false_type empty() noexcept
    {
        return {};
    }
};

template<typename T, int N>
inline __device__ void Load(Array<T, N> &dst, const T *src) {
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        (uint4 &) dst = *(const uint4 *) src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        (uint2 &) dst = *(const uint2 *) src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
        (uint1 &) dst = *(const uint1 *) src;
    } else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {
        //  uncoalesced
        // static_assert(bitsof<T> % 8 == 0, "raw pointer arithmetic of sub-byte types");
        constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
        for (int i = 0; i < M; ++i) {
            *((uint4 *) &dst + i) = *((uint4 *) src + i);
        }
    } else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename T, int N>
inline __device__ void Store(T *dst, const Array<T, N> &src) {
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        *(uint4 *) dst = (const uint4 &) src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        *(uint2 *) dst = (const uint2 &) src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint1)) {
        *(uint1 *) dst = (const uint1 &) src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(ushort)) {
        *(ushort *) dst = (const ushort &) src;
    } else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {
        //  uncoalesced
        // static_assert(bitsof<T> % 8 == 0, "raw pointer arithmetic of sub-byte types");
        constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
        for (int i = 0; i < M; ++i) {
            *((uint4 *) dst + i) = *((uint4 *) &src + i);
        }
    } else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename T>
__device__ __forceinline__ T warp_prefix_sum(T val, int lane_id) {
    T x = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        T y = __shfl_up_sync(0xffffffff, x, offset);
        if (lane_id >= offset) {
            x += y;
        }
    }
    return x - val;
}


template<typename T, int block_dim>
__device__
T block_prefix_sum(T val) {
    int lane_id = threadIdx.x & 0x1f;
    int warp_id = threadIdx.x >> 5;
    __shared__ T shared[block_dim >> 5];
    T warp_prefix = warp_prefix_sum(val, lane_id);

    if (lane_id == 31) {
        shared[warp_id] = warp_prefix + val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        shared[threadIdx.x] = warp_prefix_sum(shared[threadIdx.x], lane_id);
    }

    __syncthreads();

    return warp_prefix + shared[warp_id];
}

template<typename T, int block_dim, int items_per_thread>
__global__
void prefix_sum(const T *__restrict__ X, T *Y, int *flags, T *scan_value, int *block_counter, size_t N) {
    __shared__ size_t sbid;

    if (threadIdx.x == 0) {
        sbid = atomicAdd(block_counter, 1);
    }

    __syncthreads();

    size_t bid = sbid;

    auto idx = bid * blockDim.x + threadIdx.x;

    Array<T, items_per_thread> vec{};
    Load(vec, X + idx * items_per_thread);

#pragma unroll
    for (int i = 1; i < items_per_thread; ++i) {
        vec[i] += vec[i - 1];
    }

    auto reduced_total_sum = block_prefix_sum<T, block_dim>(vec[items_per_thread - 1]);

    __shared__ T prev_sum;
    if (threadIdx.x == block_dim - 1) {
        while (bid != 0 && atomicAdd(&flags[bid], 0) == 0) {
        }

        prev_sum = scan_value[bid];

        scan_value[bid + 1] = prev_sum + reduced_total_sum + vec[items_per_thread - 1];

        __threadfence();
        atomicAdd(&flags[bid + 1], 1);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < items_per_thread; ++i) {
        vec[i] = reduced_total_sum + prev_sum + vec[i];
    }

    Store(Y + idx * items_per_thread, vec);
}

template
__global__
void prefix_sum<float, 128, ITEMS_PER_THREAD>(const float *__restrict__ X, float *Y, int *flags, float *scan_value,
                                              int *block_counter,
                                              size_t N);

template
__global__
void prefix_sum<float, 256, ITEMS_PER_THREAD>(const float *__restrict__ X, float *Y, int *flags, float *scan_value,
                                              int *block_counter,
                                              size_t N);

template
__global__
void prefix_sum<float, 512, ITEMS_PER_THREAD>(const float *__restrict__ X, float *Y, int *flags, float *scan_value,
                                              int *block_counter,
                                              size_t N);

template
__global__
void prefix_sum<float, 1024, ITEMS_PER_THREAD>(const float *__restrict__ X, float *Y, int *flags, float *scan_value,
                                               int *block_counter,
                                               size_t N);

void launch_prefix_sum(const float *d_X, float *d_output, int n, int *flags, float *scan_value, int *block_counter,
                       int grid_dim, int block_dim, cudaStream_t stream) {
    if (block_dim == 128) {
        prefix_sum<float, 128, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(
            d_X, d_output, flags, scan_value, block_counter, n);
    } else if (block_dim == 256) {
        prefix_sum<float, 256, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(
            d_X, d_output, flags, scan_value, block_counter, n);
    } else if (block_dim == 512) {
        prefix_sum<float, 512, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(
            d_X, d_output, flags, scan_value, block_counter, n);
    } else if (block_dim == 1024) {
        prefix_sum<float, 1024, ITEMS_PER_THREAD><<<grid_dim, block_dim, 0, stream>>>(
            d_X, d_output, flags, scan_value, block_counter, n);
    }
}

bool check_result(float *output, float *target, size_t n, float eps = 1e-1) {
    for (int i = 0; i < n; ++i) {
        if (fabs(output[i] - target[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    size_t n = FLAGS_n;
    size_t block_dim = FLAGS_block_dim;
    size_t num_warmups = FLAGS_num_warmups;
    size_t num_repeats = FLAGS_num_repeats;

    float *h_X, *h_output, *h_target;
    float *d_X, *d_output;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_X = (float *) malloc(n * sizeof(float));
    h_output = (float *) malloc(n * sizeof(float));
    h_target = (float *) malloc(n * sizeof(float));

    // Random number generator
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(-1.f, 1.f); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < n; ++i) {
        h_X[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    prefix_sum_cpu(h_X, h_target, n);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "Latency for prefix sum on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_X, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_output, n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_X, h_X, n * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    size_t grid_dim = (n + block_dim - 1) / block_dim / 4;

    int *flags;
    float *scan_value;
    int *block_counter;
    CHECK_CUDA_ERROR(cudaMalloc((void **) &flags, (grid_dim + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &scan_value, (grid_dim + 1) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &block_counter, 1 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(block_counter, 0, 1 * sizeof(int)));

    std::function<void(cudaStream_t)> bound_function_prefix_sum{
        std::bind(launch_prefix_sum, d_X, d_output, n, flags, scan_value, block_counter, grid_dim, block_dim, stream)
    };

    float const latency_gpu{measure_performance(bound_function_prefix_sum, stream, num_repeats, num_warmups)};
    std::cout << "Latency for prefix sum on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (check_result(h_output, h_target, n)) {
        std::cout << "Result is correct" << std::endl;
    } else {
        std::cout << "Result is incorrect" << std::endl;
    }

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_X);
    free(h_output);
    free(h_target);
    CHECK_CUDA_ERROR(cudaFree(d_X));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(flags));
    CHECK_CUDA_ERROR(cudaFree(scan_value));
    CHECK_CUDA_ERROR(cudaFree(block_counter));
    return 0;
}
