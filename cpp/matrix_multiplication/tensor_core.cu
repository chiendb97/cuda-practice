//
// Created by root on 1/10/25.
//

#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <string>

#include <cuda_runtime.h>
#include <mma.h>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

void check_cuda_error(cudaError_t err, const char *const func,
                      const char *const file, const int line) {
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
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename T>
bool check_result(const T *target, const T *result, int size,
                  float eps = 1e-5) {
  T *h_result = static_cast<T *>(malloc(size * sizeof(T)));
  cudaMemcpy(h_result, result, size * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    if (fabs(h_result[i] - target[i]) > eps) {
      return false;
    }
  }
  free(h_result);
  return true;
}

template <typename T> void init_random(T *a, int size) {
  // Random number generator
  std::random_device rd;  // Obtain a random number from hardware
  std::mt19937 gen(rd()); // Seed the generator
  std::uniform_int_distribution<> distr(-10, 10); // Define the range

  // Generate and print a random integer
  for (int i = 0; i < size; ++i) {
    a[i] = distr(gen);
  }
}

template <class T>
double measure_performance(const std::function<T(cudaStream_t)> &bound_function,
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

  double const latency{(double)time / num_repeats};

  return latency;
}

template <typename T, typename ACCUM>
void matrix_multiplication(int M, int N, int K, const T *A, const T *B,
                           ACCUM *C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        C[m + n * M] += (ACCUM)(A[m + k * M] * B[n + k * N]);
      }
    }
  }
}

template <typename T, typename ACCUM, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void matrix_multiplication_kernel(int M, int N, int K, const T *A,
                                             int lda, const T *B, int ldb,
                                             ACCUM *C, int ldc, float alpha,
                                             float beta) {
  uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
  uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

  // Declare the fragments.
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T,
                         WMMA_FRAG_LAYOUT_A>
      a_frag{};
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T,
                         WMMA_FRAG_LAYOUT_B>
      b_frag{};
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         ACCUM>
      acc_frag{};
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         ACCUM>
      c_frag{};

  // Make sure the accumulator starts from 0.
  nvcuda::wmma::fill_fragment(acc_frag, static_cast<ACCUM>(0));

  // Loop over K.
  for (unsigned k = 0; k < K; k += WMMA_K) {
    // Determine the first element of the mma matrices on the linear memory.
    // Matrix A mma matrix
    uint32_t const matrix_mma_a_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_a_col_idx{k};
    // Matrix B mma matrix
    uint32_t const matrix_mma_b_row_idx{warpN * WMMA_N};
    uint32_t const matrix_mma_b_col_idx{k};

    // Determine the memory address of the first element of the mma
    // matrices. Notice that all the matrices are assumed to be
    // column-major. Therefore, the indexing is different from the
    // row-major indexing that we commonly see.
    T const *matrix_mma_a_mptr{A + matrix_mma_a_row_idx +
                               matrix_mma_a_col_idx * lda};
    T const *matrix_mma_b_mptr{B + matrix_mma_b_row_idx +
                               matrix_mma_b_col_idx * ldb};
    // Load the mma matrix inputs.
    nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
    nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha.
  uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
  uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

  ACCUM *matrix_mma_c_mptr{C + matrix_mma_c_row_idx +
                           matrix_mma_c_col_idx * ldc};
  nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                 nvcuda::wmma::mem_col_major);
  // Let the compiler figure out how to do the elementwise operation.
  // Such elementwise operation can be scaling, accumulation,
  // quantization, etc.
  // https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#id40
  // Be careful when dealing with the integer types.
  for (uint32_t i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
  }
  // Store the output
  nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                  nvcuda::wmma::mem_col_major);
}

template <typename T, typename ACCUM>
void invoke_matrix_multiplication(int M, int N, int K, const T *A, const T *B,
                                  ACCUM *C, cudaStream_t stream = 0) {
  const int lda = M;
  const int ldb = N;
  const int ldc = M;

  const float alpha{1.f};
  const float beta{0.f};

  constexpr int WMMA_M{16};
  constexpr int WMMA_N{16};
  constexpr int WMMA_K{16};
  constexpr int WARP_SIZE{32};

  const int num_warps_x = 4;
  const int num_warps_y = 4;

  dim3 block_dim(num_warps_x * WARP_SIZE, num_warps_y);
  dim3 grid_dim((M + WMMA_M * num_warps_x - 1) / (WMMA_M * num_warps_x),
                (N + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y));

  matrix_multiplication_kernel<T, ACCUM, WMMA_M, WMMA_N, WMMA_K,
                               nvcuda::wmma::col_major, nvcuda::wmma::row_major>
      <<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, lda, B, ldb, C, ldc,
                                           alpha, beta);
}

int main(int argc, char **argv) {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  if (argc > 1) {
    M = std::stoi(argv[1]);
  }

  if (argc > 2) {
    N = std::stoi(argv[2]);
  }

  if (argc > 3) {
    K = std::stoi(argv[3]);
  }

  using T = half;
  using ACCUM = float;

  // Init host data
  T *h_a, *h_b;
  ACCUM *h_c;

  h_a = static_cast<T *>(malloc(M * K * sizeof(T)));
  h_b = static_cast<T *>(malloc(N * K * sizeof(T)));
  h_c = static_cast<ACCUM *>(malloc(M * N * sizeof(ACCUM)));

  init_random(h_a, M * K);
  init_random(h_b, N * K);
  memset(h_c, 0, M * N * sizeof(ACCUM));

  // Init device data
  T *d_a, *d_b;
  float *d_c;

  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_a, M * K * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b, N * K * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_c, M * N * sizeof(ACCUM)));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_a, h_a, M * K * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_b, h_b, N * K * sizeof(T), cudaMemcpyHostToDevice));

  std::clock_t time_start = std::clock();
  matrix_multiplication(M, N, K, h_a, h_b, h_c);
  std::clock_t time_end = std::clock();
  double latency_cpu = (double)(time_end - time_start) / CLOCKS_PER_SEC * 1000;
  std::cout << "Process time for matrix multiplication on CPU : " << latency_cpu
            << std::endl;

  double latency_gpu = 0;
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  CHECK_CUDA_ERROR(cudaMemset(d_c, 0, M * N * sizeof(ACCUM)));
  std::function<void(cudaStream_t)> bound_function_invoke_matrix_multiplication{
      std::bind(invoke_matrix_multiplication<T, ACCUM>, M, N, K, d_a, d_b, d_c,
                stream)};

  latency_gpu = measure_performance(bound_function_invoke_matrix_multiplication,
                                    stream, 1, 0);
  std::cout << "Process time for matrix multiplication on GPU : " << latency_gpu
            << std::endl;

  if (check_result(h_c, d_c, M * N)) {
    std::cout << "Yes" << std::endl;
  } else {
    std::cout << "Wrong answer" << std::endl;
  }

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
