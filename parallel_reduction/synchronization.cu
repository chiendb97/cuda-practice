//
// Created by chiendb on 4/21/24.
//

#include <iostream>
#include <random>
#include <ctime>


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

void sum_array_cpu(int *a, int &sum, int size) {
    for (int i = 0; i < size; ++i) {
        sum += a[i];
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

__global__
void sum_array(int *a, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (threadIdx.x % (offset * 2) == 0) {
            a[idx] += a[idx + offset];
        }
        __syncthreads();
    }
}

void launch_sum_array(int *a, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    sum_array<<<grid_dim, block_dim, 0, stream>>>(a, size);
}

bool check_result(int *h_s, const int target, int block_dim, int size) {
    int output = 0;
    for (int i = 0; i < size; i += block_dim) {
        output += h_s[i];
    }
    if (output != target) {
        std::cout << "output: " << output << ", target: " << target << "\n";
    }
    return output == target;
}

int main() {
    int size = 1 << 22;
    int *h_a, *h_s, sum = 0;
    int *d_a;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_a = (int *) malloc(size * sizeof(int));
    h_s = (int *) malloc(size * sizeof(int));

    // Random number generator
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(1, 10); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < size; ++i) {
        h_a[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    sum_array_cpu(h_a, sum, size);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC;
    std::cout << "Latency for sum array on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_a, size * sizeof(int)));

    for (int block_dim = 64; block_dim <= 512; block_dim *= 2) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        int grid_dim = size / block_dim;
        std::function<void(cudaStream_t)> bound_function_sum_array{
                std::bind(launch_sum_array, d_a, size, grid_dim, block_dim, stream)};

        float const latency_gpu{measure_performance(bound_function_sum_array, stream, 1, 0)};
        std::cout << "Latency for sum array on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_s, d_a, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        bool success = check_result(h_s, sum, block_dim, size);
        if (success) {
            std::cout << "Result is correct\n";
        } else {
            std::cout << "Result is incorrect\n";
        }
    }

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_a);
    free(h_s);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    return 0;
}