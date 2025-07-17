//
// Created by chiendb on 3/4/24.
//

#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <gflags/gflags.h>

DEFINE_int32(kernel_size, 7, "kernel_size");
DEFINE_int32(stride, 3, "stride");
DEFINE_int32(padding, 2, "padding");
DEFINE_int32(H, 1 << 20, "H");
DEFINE_uint32(block_dim, 256, "block_dim");
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

void average_pooling_1d_cpu(const float *X, float *Y, int kernel_size, int padding, int stride, int H, int H_out) {
    for (int i = 0; i < H_out; ++i) {
        float sum = 0;
        for (int m = 0; m < kernel_size; ++m) {
            if (i * stride + m - padding >= 0 && i * stride + m - padding < H) {
                sum += X[i * stride + m - padding];
            }
        }

        Y[i] = sum / kernel_size;
    }
}


__global__
void average_pooling_1d(const float *__restrict__ X, float *Y, int kernel_size, int padding, int stride,
                        int H, int H_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= H_out) {
        return;
    }

    float sum = 0;
    int start = max(idx * stride - padding, 0);
    int end = min(idx * stride + kernel_size - 1 - padding, H - 1);
    for (auto i = start; i <= end; ++i) {
        sum += X[i];
    }

    Y[idx] = sum / kernel_size;
}


void launch_average_pooling_1d(const float *d_X, float *d_output, int kernel_size, int padding, int stride,
                               int H, int H_out, int grid_dim, int block_dim, cudaStream_t stream) {
    average_pooling_1d<<<grid_dim, block_dim, 0, stream>>>(d_X, d_output, kernel_size, padding, stride, H, H_out);
}

bool check_result(float *output, float *target, size_t n, float eps = 1e-2) {
    for (int i = 0; i < n; ++i) {
        if (fabs(output[i] - target[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto kernel_size = FLAGS_kernel_size;
    auto stride = FLAGS_stride;
    auto padding = FLAGS_padding;
    auto H = FLAGS_H;
    size_t block_dim = FLAGS_block_dim;
    size_t num_warmups = FLAGS_num_warmups;
    size_t num_repeats = FLAGS_num_repeats;

    auto H_out = (H + 2 * padding - kernel_size) / stride + 1;

    float *h_X, *h_output, *h_target;
    float *d_X, *d_output;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_X = (float *) malloc(H * sizeof(float));
    h_output = (float *) malloc(H_out * sizeof(float));
    h_target = (float *) malloc(H_out * sizeof(float));

    // Random number generator
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(0.f, 1.f); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < H; ++i) {
        h_X[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    average_pooling_1d_cpu(h_X, h_target, kernel_size, padding, stride, H, H_out);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "Latency for average pooling 1d on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_X, H * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_output, H_out * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_X, h_X, H * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    size_t grid_dim = (H_out + block_dim - 1) / block_dim;
    std::function<void(cudaStream_t)> bound_function_average_pooling_1d{
        std::bind(launch_average_pooling_1d, d_X, d_output, kernel_size, padding, stride, H, H_out, grid_dim, block_dim,
                  stream)
    };

    float const latency_gpu{measure_performance(bound_function_average_pooling_1d, stream, num_repeats, num_warmups)};
    std::cout << "Latency for average pooling 1d on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, H_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    bool success = check_result(h_output, h_target, H_out);
    if (success) {
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
