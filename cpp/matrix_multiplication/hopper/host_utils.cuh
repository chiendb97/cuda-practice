#pragma once
#include <functional>
#include <random>
#include <thrust/system/cuda/detail/util.h>
#include <cublas_v2.h>

#include "params.cuh"

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

inline void check_cuda_error(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "Cuda Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

inline void check_last_cuda_error(const char *const file, const int line)
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

#define CHECK_CUBLAS_ERROR(val) check_cublas_error((val), #val, __FILE__, __LINE__)

inline void check_cublas_error(cublasStatus_t err, const char *const func, const char *const file, const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Cuda Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << "CUBLAS_STATUS_ERROR " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void init_matrix(T *matrix, size_t row, size_t col)
{
    // Random number generator
    std::random_device rd;                                 // Obtain a random number from hardware
    std::mt19937 gen(rd());                                // Seed the generator
    std::uniform_real_distribution<> distr(-0.01f, 0.01f); // Define the range

    // Generate and print a random integer
    for (int i = 0; i < row * col; ++i)
    {
        matrix[i] = (T)distr(gen);
    }
}

template <typename T>
std::pair<MatMulParams<T>, MatMulParams<T>> setup_params(size_t M, size_t N, size_t K, float alpha, float beta,
                                                         cudaStream_t stream = 0)
{
    MatMulParams<T> host_params(M, N, K, alpha, beta);
    MatMulParams<T> device_params(M, N, K, alpha, beta);

    host_params.A = (T *)malloc(M * K * sizeof(T));
    host_params.B = (T *)malloc(N * K * sizeof(T));
    host_params.C = (T *)malloc(M * N * sizeof(T));
    host_params.D = (T *)malloc(M * N * sizeof(T));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&device_params.A, M * K * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&device_params.B, N * K * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&device_params.C, M * N * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&device_params.D, M * N * sizeof(T)));

    init_matrix(host_params.A, host_params.M, host_params.K);
    init_matrix(host_params.B, host_params.N, host_params.K);
    init_matrix(host_params.C, host_params.M, host_params.N);
    memset(host_params.D, 0, M * N * sizeof(T));

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(device_params.A, host_params.A, M * K * sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(device_params.B, host_params.B, N * K * sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(device_params.C, host_params.C, M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(device_params.D, host_params.D, M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    return std::make_pair(host_params, device_params);
}

template <typename T>
void free_params(MatMulParams<T> &host_params, MatMulParams<T> &device_params)
{
    free(host_params.A);
    free(host_params.B);
    free(host_params.C);
    free(host_params.D);

    CHECK_CUDA_ERROR(cudaFree(device_params.A));
    CHECK_CUDA_ERROR(cudaFree(device_params.B));
    CHECK_CUDA_ERROR(cudaFree(device_params.C));
    CHECK_CUDA_ERROR(cudaFree(device_params.D));
}

template <typename T>
void matrix_multiplication_cpu(size_t M, size_t N, size_t K, float alpha, const T *A, const T *B, float beta,
                               const T *C, T *D)
{
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            T accum = T(0);
            for (int k = 0; k < K; ++k)
            {
                accum += A[m * K + k] * B[n * K + k];
            }
            D[m * N + n] = (T)alpha * accum + (T)beta * C[m * N + n];
        }
    }
}

template <typename T>
bool check_result(T *device_output, T *host_target, uint32_t m, uint32_t n, float eps = 1e-2)
{
    T *host_output = (T *)malloc(m * n * sizeof(T));

    CHECK_CUDA_ERROR(cudaMemcpy(host_output, device_output, m * n * sizeof(T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m * n; ++i)
    {
        float output = __half2float(host_output[i]);
        float target = __half2float(host_target[i]);

        if (fabs(output - target) > eps)
        {
            std::cout << "Error, " << "Output: " << output << ", Target: " << target << std::endl;
            return false;
        }
    }

    free(host_output);
    return true;
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


template <uint32_t SMEM_HEIGHT, uint32_t SMEM_WIDTH>
void __create_tensor_map_host(half *tensor_ptr, uint32_t gmem_height, uint32_t gmem_width, CUtensorMap *tensor_map)
{
    constexpr uint32_t rank = 2;
    uint64_t gmem_size[2] = {uint64_t{gmem_width}, uint64_t{gmem_height}};
    uint64_t gmem_stride[1] = {sizeof(half) * gmem_width};

    constexpr uint32_t smem_size[2] = {SMEM_WIDTH, SMEM_HEIGHT};
    constexpr uint32_t smem_stride[2] = {1, 1}; // ???

    CUresult res = cuTensorMapEncodeTiled(
        tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16, // CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        rank,
        tensor_ptr,
        gmem_size,
        gmem_stride,
        smem_size,
        smem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS)
    {
        std::cout << "Error encoding tensor map" << std::endl;
    }

    assert(res == CUDA_SUCCESS);
}

template <uint32_t SMEM_HEIGHT, uint32_t SMEM_WIDTH>
void create_tensor_map(half *tensor_ptr, uint32_t gmem_height, uint32_t gmem_width, CUtensorMap *tensor_map_device)
{
    CUtensorMap tensor_map_host;
    __create_tensor_map_host<SMEM_HEIGHT, SMEM_WIDTH>(tensor_ptr, gmem_height, gmem_width, &tensor_map_host);
    CHECK_CUDA_ERROR(cudaMemcpy(tensor_map_device, &tensor_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
}
