//
// Created by chiendb on 3/4/24.
//

#include <iostream>
#include <random>
#include<ctime>


#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "Cuda Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)
void check_last_cuda_error(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void sum_array_cpu(int *a, int *b, int *c, int *s, int size) {
    for (int i = 0; i < size; ++i) {
        s[i] = a[i] + b[i] + c[i];
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0}; i < num_repeats; ++i)
    {
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
void sum_array(int *a, int *b, int *c, int *s, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        s[i] = a[i] + b[i] + c[i];
    }
}

void launch_sum_array(int *a, int *b, int *c, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    sum_array<<<grid_dim, block_dim, 0, stream>>>(a, b, c, s, size);
}

bool check_result(int *output, int *target, int size) {
    for (int i = 0; i < size; ++i) {
        if (output[i] != target[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    int size = 1 << 22;
    int *h_a, *h_b, *h_c, *h_s_target, *h_s_output;
    int *d_a, *d_b, *d_c, *d_s;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_a = (int*) malloc(size * sizeof(int));
    h_b = (int*) malloc(size * sizeof(int));
    h_c = (int*) malloc(size * sizeof(int));
    h_s_target = (int*) malloc(size * sizeof(int));
    h_s_output = (int*) malloc(size * sizeof(int));

    // Random number generator
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(1, 100000); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < size; ++i) {
        h_a[i] = distr(gen);
        h_b[i] = distr(gen);
        h_c[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    sum_array_cpu(h_a, h_b, h_c, h_s_target, size);
    std::clock_t time_end = std::clock();

    double latency_cpu =  (double) (time_end - time_start) / CLOCKS_PER_SEC;
    std::cout << "Latency for sum array on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_a, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_b, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_c, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_s, size * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_c, h_c, size * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (int block_dim = 64; block_dim <= 512; block_dim *= 2) {
        std::function<void(cudaStream_t)> bound_function_sum_array{
                std::bind(launch_sum_array, d_a, d_b, d_c, d_s, size, 1, block_dim, stream)};

        float const latency_gpu{measure_performance(bound_function_sum_array, stream)};
        std::cout << "Latency for sum array on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_s_output, d_s, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    bool success = check_result(h_s_output, h_s_target, size);
    if (success) {
        std::cout << "Result is correct";
    } else {
        std::cout << "Result is incorrect";
    }

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_s_output);
    free(h_s_target);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaFree(d_s));
    return 0;
}