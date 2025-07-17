//
// Created by chiendb on 11/03/24.
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

void transpose_cpu(int *source, int *target, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            target[j * M + i] = source[i * N + j];
        }
    }
}

template<class T>
double measure_performance(std::function<T(cudaStream_t)> bound_function,
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

    double const latency{(double) time / num_repeats};

    return latency;
}

double measure_memory_bandwidth(int M, int N, double latency) {
    return 2e-6 * M * N * 4 / latency;
}

__global__
void copy_row(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) {
        target[iy * M + ix] = source[iy * M + ix];
    }
}

__global__
void copy_column(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) {
        target[ix * N + iy] = source[ix * N + iy];
    }
}

__global__
void transpose_row(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) {
        target[ix * N + iy] = source[iy * M + ix];
    }
}

__global__
void transpose_column(int *source, int *target, int M, int N) {
    auto ix = blockDim.x * blockIdx.x + threadIdx.x;
    auto iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < M && iy < N) {
        target[iy * M + ix] = source[ix * N + iy];
    }
}

__global__
void transpose_diagonal_row(int *source, int *target, int M, int N) {
    auto block_idx_y = blockIdx.x;
    auto block_idx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    auto ix = blockDim.x * block_idx_x + threadIdx.x;
    auto iy = blockDim.y * block_idx_y + threadIdx.y;
    if (ix < M && iy < N) {
        target[ix * N + iy] = source[iy * M + ix];
    }
}

__global__
void transpose_diagonal_column(int *source, int *target, int M, int N) {
    auto block_idx_y = blockIdx.x;
    auto block_idx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    auto ix = blockDim.x * block_idx_x + threadIdx.x;
    auto iy = blockDim.y * block_idx_y + threadIdx.y;
    if (ix < M && iy < N) {
        target[iy * M + ix] = source[ix * N + iy];
    }
}

void launch_copy_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    copy_row<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}

void launch_copy_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    copy_column<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}

void launch_transpose_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_row<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}

void launch_transpose_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_column<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}

void launch_transpose_diagonal_row(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_diagonal_row<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}

void launch_transpose_diagonal_column(int *source, int *target, int M, int N, dim3 grid_dim, dim3 block_dim, cudaStream_t stream) {
    transpose_diagonal_column<<<grid_dim, block_dim, 0, stream>>>(source, target, M, N);
}

bool check_result(int *source, int *target, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (source[i * N + j] != target[j * M + i]) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int kernel_type = 0;
    int M = 2048;
    int N = 2048;
    int bM = 16;
    int bN = 16;
    int *h_source, *h_target;
    int *d_source, *d_target;

    if (argc > 1) {
        kernel_type = std::stoi(argv[1]);
    }

    if (argc > 2) {
        bM = std::stoi(argv[2]);
    }

    if (argc > 3) {
        bN = std::stoi(argv[3]);
    }

    if (argc > 4) {
        M = std::stoi(argv[4]);
    }

    if (argc > 5) {
        N = std::stoi(argv[5]);
    }

    auto size = M * N;

    dim3 block_dim(bM, bN);
    dim3 grid_dim((M + bM - 1) / bM, (N + bN - 1) / bN);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    h_source = (int *) malloc(size * sizeof(int));
    h_target = (int *) malloc(size * sizeof(int));

    // Random number generator
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(1, 1000000000); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < size; ++i) {
        h_source[i] = distr(gen);
    }

    std::clock_t time_start = std::clock();
    transpose_cpu(h_source, h_target, M, N);
    std::clock_t time_end = std::clock();

    double latency_cpu = (double) (time_end - time_start) / CLOCKS_PER_SEC * 1000;
    auto memory_bandwidth_cpu = measure_memory_bandwidth(M, N, latency_cpu);
    std::cout << "Memory bandwidth for transpose on CPU : " << memory_bandwidth_cpu << std::endl;

    cudaSetDevice(0);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_source, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_target, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_source, h_source, size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    double latency_gpu = 0;
    double memory_bandwidth = 0;
    int num_repeats = 100;
    int num_warmups = 10;

    if (kernel_type == 0 || kernel_type == 1) {
        CHECK_CUDA_ERROR(cudaMemset(d_target, 0, size * sizeof(int)));
        std::function<void(cudaStream_t)> bound_function_copy_row{
            std::bind(launch_copy_row, d_source, d_target, M, N, grid_dim, block_dim, stream)};
        latency_gpu = measure_performance(bound_function_copy_row, stream, num_repeats, num_warmups);
        memory_bandwidth = measure_memory_bandwidth(M, N, latency_gpu);
        std::cout << "Memory bandwidth for copy row on GPU : " << memory_bandwidth << std::endl;
    }

    if (kernel_type == 0 || kernel_type == 2) {
        CHECK_CUDA_ERROR(cudaMemset(d_target, 0, size * sizeof(int)));
        std::function<void(cudaStream_t)> bound_function_copy_column{
            std::bind(launch_copy_column, d_source, d_target, M, N, grid_dim, block_dim, stream)};
        latency_gpu = measure_performance(bound_function_copy_column, stream, num_repeats, num_warmups);
        memory_bandwidth = measure_memory_bandwidth(M, N, latency_gpu);
        std::cout << "Memory bandwidth for copy column on GPU : " << memory_bandwidth << std::endl;
    }

    if (kernel_type == 0 || kernel_type == 3) {
        CHECK_CUDA_ERROR(cudaMemset(d_target, 0, size * sizeof(int)));
        std::function<void(cudaStream_t)> bound_function_transpose_row{
            std::bind(launch_transpose_row, d_source, d_target, M, N, grid_dim, block_dim, stream)
        };
        latency_gpu = measure_performance(bound_function_transpose_row, stream, num_repeats, num_warmups);
        memory_bandwidth = measure_memory_bandwidth(M, N, latency_gpu);
        std::cout << "Memory bandwidth for transpose row on GPU : " << memory_bandwidth << std::endl;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_target, d_target, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (!check_result(h_source, h_target, M, N)) {
            std::cout << "Result is incorrect\n";
        }
    }

    if (kernel_type == 0 || kernel_type == 4) {
        CHECK_CUDA_ERROR(cudaMemset(d_target, 0, size * sizeof(int)));
        std::function<void(cudaStream_t)> bound_function_transpose_column{
            std::bind(launch_transpose_column, d_source, d_target, M, N, grid_dim, block_dim, stream)
        };
        latency_gpu = measure_performance(bound_function_transpose_column, stream, num_repeats, num_warmups);
        memory_bandwidth = measure_memory_bandwidth(M, N, latency_gpu);
        std::cout << "Memory bandwidth for transpose column on GPU : " << memory_bandwidth << std::endl;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_target, d_target, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (!check_result(h_source, h_target, M, N)) {
            std::cout << "Result is incorrect\n";
        }
    }

    if (bM == bN) {
        if (kernel_type == 0 || kernel_type == 5) {
            CHECK_CUDA_ERROR(cudaMemset(d_target, 0, size * sizeof(int)));
            std::function<void(cudaStream_t)> bound_function_transpose_diagonal_row{
                std::bind(launch_transpose_diagonal_row, d_source, d_target, M, N, grid_dim, block_dim, stream)
            };
            latency_gpu = measure_performance(bound_function_transpose_diagonal_row, stream, num_repeats, num_warmups);
            memory_bandwidth = measure_memory_bandwidth(M, N, latency_gpu);
            std::cout << "Memory bandwidth for transpose diagonal row on GPU : " << memory_bandwidth << std::endl;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_target, d_target, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            if (!check_result(h_source, h_target, M, N)) {
                std::cout << "Result is incorrect\n";
            }
        }

        if (kernel_type == 0 || kernel_type == 6) {
            CHECK_CUDA_ERROR(cudaMemset(d_target, 0, size * sizeof(int)));
            std::function<void(cudaStream_t)> bound_function_transpose_diagonal_column{
                std::bind(launch_transpose_diagonal_column, d_source, d_target, M, N, grid_dim, block_dim, stream)
            };
            latency_gpu = measure_performance(bound_function_transpose_diagonal_column, stream, num_repeats,
                                              num_warmups);
            memory_bandwidth = measure_memory_bandwidth(M, N, latency_gpu);
            std::cout << "Memory bandwidth for transpose diagonal column on GPU : " << memory_bandwidth << std::endl;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_target, d_target, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            if (!check_result(h_source, h_target, M, N)) {
                std::cout << "Result is incorrect\n";
            }
        }
    }

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(h_source);
    free(h_target);
    CHECK_CUDA_ERROR(cudaFree(d_source));
    CHECK_CUDA_ERROR(cudaFree(d_target));
    return 0;
}
