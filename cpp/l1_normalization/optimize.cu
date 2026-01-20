//
// L1 normalization optimized (warp-reduction)
//
#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <gflags/gflags.h>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
inline void check_cuda_error(cudaError_t err, const char *const func, const char *const file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "Cuda Runtime Error at: " << file << ":" << line << "\n"
                  << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)
inline void check_last_cuda_error(const char *const file, int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << "\n"
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

DEFINE_uint32(b, 4, "b");
DEFINE_uint32(n, 8192, "n (must be divisible by 4)");
DEFINE_uint32(block_dim, 256, "block dim");
DEFINE_uint32(num_warmups, 1, "num_warmups");
DEFINE_uint32(num_repeats, 1, "num_repeats");

template<class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function, cudaStream_t stream,
                          unsigned int num_repeats = 100, unsigned int num_warmups = 100) {
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

    return time / num_repeats;
}

void l1_norm_cpu(const float *X, float *Y, size_t b, size_t n, float eps) {
    for (size_t i = 0; i < b; ++i) {
        float sum = 0.f;
        for (size_t j = 0; j < n; ++j) {
            sum += fabsf(X[i * n + j]);
        }
        float inv_norm = 1.0f / (sum + eps);
        for (size_t j = 0; j < n; ++j) {
            Y[i * n + j] = X[i * n + j] * inv_norm;
        }
    }
}

__forceinline__ __device__ void accumulate_abs(const float4 &a, float &sum) {
    sum += fabsf(a.x) + fabsf(a.y) + fabsf(a.z) + fabsf(a.w);
}

__forceinline__ __device__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int block_size>
__launch_bounds__(block_size, 2)
__global__ void l1_norm_opt(const float4 *__restrict__ X, float4 *Y, size_t B, size_t N, float eps) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x & (warp_size - 1);
    const int warp_id = threadIdx.x / warp_size;
    const int num_warps = block_size / warp_size;

    const auto ti = blockIdx.x;
    if (ti >= B) {
        return;
    }

    X += ti * N;
    float sum = 0.f;
    for (size_t i = threadIdx.x; i < N; i += block_size) {
        float4 vec = X[i];
        accumulate_abs(vec, sum);
    }

    sum = warp_sum(sum);

    __shared__ float warp_partials[block_size / warp_size];
    if (lane == 0) {
        warp_partials[warp_id] = sum;
    }
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
        vec.x *= inv_norm;
        vec.y *= inv_norm;
        vec.z *= inv_norm;
        vec.w *= inv_norm;
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

    if (block_dim == 128) {
        l1_norm_opt<128><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    } else if (block_dim == 256) {
        l1_norm_opt<256><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    } else if (block_dim == 512) {
        l1_norm_opt<512><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    } else if (block_dim == 1024) {
        l1_norm_opt<1024><<<grid_dim, block_dim, 0, stream>>>(X4, Y4, b, n4, eps);
    } else {
        printf("Unsupported block dim: %d\n", block_dim);
    }
}

bool check_result(float *output, float *target, size_t n, float eps = 1e-2f) {
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(output[i] - target[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    size_t b = FLAGS_b;
    size_t n = FLAGS_n;
    size_t block_dim = FLAGS_block_dim;
    size_t num_warmups = FLAGS_num_warmups;
    size_t num_repeats = FLAGS_num_repeats;
    const float eps = 1e-10f;

    if (n % 4 != 0) {
        std::cerr << "n must be divisible by 4 for float4 access\n";
        return 1;
    }

    float *h_X = static_cast<float *>(malloc(b * n * sizeof(float)));
    float *h_output = static_cast<float *>(malloc(b * n * sizeof(float)));
    float *h_target = static_cast<float *>(malloc(b * n * sizeof(float)));

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> distr(0.f, 1.f);
    for (size_t i = 0; i < b * n; ++i) {
        h_X[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    l1_norm_cpu(h_X, h_target, b, n, eps);
    std::clock_t time_end = std::clock();
    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "Latency for l1 norm on CPU: " << latency_cpu << " ms\n";

    cudaSetDevice(0);
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    float *d_X;
    float *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_X, b * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_output, b * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_X, h_X, b * n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    size_t grid_dim = b;
    std::function<void(cudaStream_t)> bound_function{
            std::bind(launch_l1_norm_opt, d_X, d_output, b, n, eps, grid_dim, block_dim, stream)
    };
    float latency_gpu = measure_performance(bound_function, stream, num_repeats, num_warmups);
    std::cout << "Latency for l1 norm on GPU (opt), block_dim " << block_dim << ": " << latency_gpu << " ms\n";

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, b * n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    bool success = check_result(h_output, h_target, b * n);
    std::cout << (success ? "Result is correct" : "Result is incorrect") << std::endl;

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_X);
    free(h_output);
    free(h_target);
    CHECK_CUDA_ERROR(cudaFree(d_X));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    return 0;
}
