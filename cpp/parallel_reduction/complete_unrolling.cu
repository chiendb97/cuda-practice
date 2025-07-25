//
// Created by chiendb on 10/29/24.
//

#include <functional>
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

void reduce_cpu(int *a, int &sum, int size) {
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

    float const latency{time / 1000 / num_repeats};

    return latency;
}

__global__
void reduce_complete_unrolling(int *a, int *s, int size) {
    auto gid = blockDim.x * blockIdx.x * 8 + threadIdx.x;

    auto *idata = a + blockDim.x * blockIdx.x * 8;

    if (gid + 7 * blockDim.x < size) {
        int a1 = a[gid];
        int a2 = a[gid + blockDim.x];
        int a3 = a[gid + 2 * blockDim.x];
        int a4 = a[gid + 3 * blockDim.x];
        int a5 = a[gid + 4 * blockDim.x];
        int a6 = a[gid + 5 * blockDim.x];
        int a7 = a[gid + 6 * blockDim.x];
        int a8 = a[gid + 7 * blockDim.x];
        a[gid] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }

    __syncthreads();

    if (blockDim.x >= 1024 && threadIdx.x < 512) {
        idata[threadIdx.x] += idata[threadIdx.x + 512];
    }
    __syncthreads();

    if (blockDim.x >= 512 && threadIdx.x < 256) {
        idata[threadIdx.x] += idata[threadIdx.x + 256];
    }
    __syncthreads();

    if (blockDim.x >= 256 && threadIdx.x < 128) {
        idata[threadIdx.x] += idata[threadIdx.x + 128];
    }
    __syncthreads();

    if (blockDim.x >= 128 && threadIdx.x < 64) {
        idata[threadIdx.x] += idata[threadIdx.x + 64];
    }
    __syncthreads();


    if (threadIdx.x < 32) {
        volatile int *vsmem = idata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        s[blockIdx.x] = idata[0];
    }
}

void launch_reduce_complete_unrolling(int *a, int *s, int size, int grid_dim, int block_dim, cudaStream_t stream) {
    reduce_complete_unrolling<<<grid_dim, block_dim, 0, stream>>>(a, s, size);
}


bool check_result(int *h_s, const int target, int grid_dim) {
    int output = 0;
    for (int i = 0; i < grid_dim; ++i) {
        output += h_s[i];
    }
    if (output != target) {
        std::cout << "output: " << output << ", target: " << target << "\n";
    }
    return output == target;
}

int main(int argc, char **argv) {
    int size = 1 << 24;
    int *h_a, *h_s, sum = 0;
    int *d_a, *d_s;

    int block_dim = std::stoi(argv[1]);
    int grid_dim = (size + block_dim - 1) / block_dim;

    grid_dim = (grid_dim + 8 - 1) / 8;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_a = (int *) malloc(size * sizeof(int));
    h_s = (int *) malloc(grid_dim * sizeof(int));

    // Random number generator
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(1, 10); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < size; ++i) {
        h_a[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    reduce_cpu(h_a, sum, size);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC;
    std::cout << "Latency for sum array on CPU: " << latency_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_a, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_s, grid_dim * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemset(d_s, 0, grid_dim * sizeof(int)));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    std::function<void(cudaStream_t)> bound_function_reduce_complete_unrolling{
            std::bind(launch_reduce_complete_unrolling, d_a, d_s, size, grid_dim, block_dim, stream)};

    float const latency_gpu{measure_performance(bound_function_reduce_complete_unrolling, stream, 1, 0)};
    std::cout << "Latency for sum array on GPU, block_dim " << block_dim << ": " << latency_gpu << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_s, d_s, grid_dim * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    bool success = check_result(h_s, sum, grid_dim);
    if (success) {
        std::cout << "Result is correct\n";
    } else {
        std::cout << "Result is incorrect\n";
    }

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_a);
    free(h_s);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_s));
    return 0;
}
