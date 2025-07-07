//
// Created by chiendb on 3/4/24.
//

#include <iostream>
#include <random>
#include <ctime>
#include <gflags/gflags.h>


DEFINE_uint32(n, 1 << 10, "n");
DEFINE_uint32(m, 1 << 10, "m");
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

void relu_cpu(const float *X, float *Y, size_t n, size_t m) {
    for (int i = 0; i < n * m; ++i) {
        Y[i] = max(X[i], 0.0f);
    }
}


__global__
void relu(const float4 *__restrict__ X, float4 *Y, size_t N) {
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
    relu<<<grid_dim, block_dim, 0, stream>>>(reinterpret_cast<const float4 *>(d_X),
                                             reinterpret_cast<float4 *>(d_output), n * m);
}

bool check_result(float *output, float *target, size_t n, float eps = 1e-4) {
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
    size_t m = FLAGS_m;
    size_t block_dim = FLAGS_block_dim;
    size_t num_warmups = FLAGS_num_warmups;
    size_t num_repeats = FLAGS_num_repeats;

    float *h_X, *h_output, *h_target;
    float *d_X, *d_output;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_X = (float *) malloc(n * m * sizeof(float));
    h_output = (float *) malloc(n * m * sizeof(float));
    h_target = (float *) malloc(n * m * sizeof(float));

    // Random number generator
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(-1.f, 1.f); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < n * m; ++i) {
        h_X[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    relu_cpu(h_X, h_target, n, m);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "Latency for relu on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_X, n * m * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_output, n * m * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_X, h_X, n * m * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    size_t grid_dim = (n * m + block_dim - 1) / block_dim / 4;

    std::function<void(cudaStream_t)> bound_function_relu{
        std::bind(launch_relu, d_X, d_output, n, m, grid_dim, block_dim, stream)
    };

    float const latency_gpu{measure_performance(bound_function_relu, stream, num_repeats, num_warmups)};
    std::cout << "Latency for relu on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, n * m * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (check_result(h_output, h_target, n * m)) {
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
    return 0;
}
