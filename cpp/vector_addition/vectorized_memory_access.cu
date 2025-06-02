//
// Created by chiendb on 3/4/24.
//

#include <iostream>
#include <random>
#include<ctime>


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


void vector_add_cpu(const float *h_input1, const float *h_input2, float *h_output, size_t n) {
    for (int i = 0; i < n; ++i) {
        h_output[i] = h_input1[i] + h_input2[i];
    }
}


__global__
void vector_add(const float4 *d_input1, const float4 *d_input2, float4 *d_output, size_t n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float4 input1 = d_input1[idx];
        const float4 input2 = d_input2[idx];;
        float4 output;
        output.x = input1.x + input2.x;
        output.y = input1.y + input2.y;
        output.z = input1.z + input2.z;
        output.w = input1.w + input2.w;
        d_output[idx] = output;
    }
}

void launch_vector_add(const float *d_input1, const float *d_input2, float *d_output, size_t n, int grid_dim,
                      int block_dim, cudaStream_t stream) {
    vector_add<<<grid_dim, block_dim, 0, stream>>>(reinterpret_cast<const float4 *>(d_input1), reinterpret_cast<const float4 *>(d_input2), reinterpret_cast<float4 *>(d_output), n / 4);
}

bool check_result(float *output, float *target, int n) {
    for (int i = 0; i < n; ++i) {
        if (output[i] != target[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    int n = 1 << 22;
    float *h_input1, *h_input2, *h_output, *h_target;
    float *d_input1, *d_input2, *d_output;

    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_input1 = (float *) malloc(n * sizeof(float));
    h_input2 = (float *) malloc(n * sizeof(float));
    h_output = (float *) malloc(n * sizeof(float));
    h_target = (float *) malloc(n * sizeof(float));

    // Random number generator
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(-100.f, 100.f); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < n; ++i) {
        h_input1[i] = distr(gen);
        h_input2[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    vector_add_cpu(h_input1, h_input2, h_target, n);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "Latency for vector add on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_input1, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_input2, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_output, n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input1, h_input1, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input2, h_input2, n * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (int block_dim = 64; block_dim <= 512; block_dim *= 2) {
        int grid_dim = (n + block_dim - 1) / block_dim / 4;
        std::function<void(cudaStream_t)> bound_function_vector_add{
            std::bind(launch_vector_add, d_input1, d_input2, d_output, n, grid_dim, block_dim, stream)
        };

        float const latency_gpu{measure_performance(bound_function_vector_add, stream)};
        std::cout << "Latency for vector add on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    bool success = check_result(h_output, h_target, n);
    if (success) {
        std::cout << "Result is correct";
    } else {
        std::cout << "Result is incorrect";
    }

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_input1);
    free(h_input2);
    free(h_output);
    free(h_target);
    CHECK_CUDA_ERROR(cudaFree(d_input1));
    CHECK_CUDA_ERROR(cudaFree(d_input2));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    return 0;
}
