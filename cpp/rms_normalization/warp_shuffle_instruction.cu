//
// Created by chiendb on 3/4/24.
//

#include <ctime>
#include <functional>
#include <gflags/gflags.h>
#include <iostream>
#include <random>

#include <cub/block/block_reduce.cuh>

DEFINE_uint32(b, 4, "b");
DEFINE_uint32(n, 8192, "n");
DEFINE_uint32(block_dim, 256, "block dim");
DEFINE_uint32(num_warmups, 1, "num_warmups");
DEFINE_uint32(num_repeats, 1, "num_repeats");

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

template <class T>
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

void rms_norm_cpu(const float *X, float *Y, size_t b, size_t n, float eps) {
  for (int i = 0; i < b; ++i) {
    float sum = 0;
    for (int j = 0; j < n; ++j) {
      sum += X[i * n + j] * X[i * n + j];
    }

    float inv_rms = 1.0f / sqrt(sum / n + eps);

    for (int j = 0; j < n; ++j) {
      Y[i * n + j] = X[i * n + j] * inv_rms;
    }
  }
}

__forceinline__ __device__ void
multiply_accumulate(const float4 &a, const float4 &b, float &sum) {
  sum += a.x * b.x;
  sum += a.y * b.y;
  sum += a.z * b.z;
  sum += a.w * b.w;
}

template <typename T, int NUM>
__forceinline__ __device__ T warp_reduce_sum(T *val) {
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
    }
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__forceinline__ __device__ T block_reduce_sum(T *val) {
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  warp_reduce_sum<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; ++i) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warp_reduce_sum<T, NUM>(val);

  return (T)(0.0f);
}

__global__ void rms_norm(const float4 *__restrict__ X, float4 *Y, size_t B,
                         size_t N, float invN, float eps) {
  auto ti = blockIdx.x;
  auto di = threadIdx.x;

  if (ti >= B) {
    return;
  }

  X += ti * N;
  float sum[1] = {0};

  float4 vec;

  // #pragma unroll 4
  for (auto i = di; i < N; i += blockDim.x) {
    vec = __ldg(&X[i]);
    sum[0] += vec.x * vec.x;
    sum[0] += vec.y * vec.y;
    sum[0] += vec.z * vec.z;
    sum[0] += vec.w * vec.w;
  }

  if (blockDim.x < 32) {
    warp_reduce_sum<float, 1>(sum);
  } else {
    block_reduce_sum<float, 1>(sum);
  }

  __shared__ float shared_inv_rms;

  if (threadIdx.x == 0) {
    shared_inv_rms = rsqrtf(sum[0] * invN + eps);
  }

  __syncthreads();

  float inv_rms = shared_inv_rms;
  Y += ti * N;

  // #pragma unroll 4
  for (auto i = di; i < N; i += blockDim.x) {
    vec = __ldg(&X[i]);
    vec.x *= inv_rms;
    vec.y *= inv_rms;
    vec.z *= inv_rms;
    vec.w *= inv_rms;
    Y[i] = vec;
  }
}

void launch_rms_norm(const float *d_X, float *d_output, size_t b, size_t n,
                     float eps, int grid_dim, int block_dim,
                     cudaStream_t stream) {
  rms_norm<<<grid_dim, block_dim, 0, stream>>>(
      reinterpret_cast<const float4 *>(d_X),
      reinterpret_cast<float4 *>(d_output), b, n / 4, 1.0f / n, eps);
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

  size_t b = FLAGS_b;
  size_t n = FLAGS_n;
  size_t block_dim = FLAGS_block_dim;
  size_t num_warmups = FLAGS_num_warmups;
  size_t num_repeats = FLAGS_num_repeats;
  const float eps = 1e-5;

  float *h_X, *h_output, *h_target;
  float *d_X, *d_output;

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  h_X = (float *)malloc(b * n * sizeof(float));
  h_output = (float *)malloc(b * n * sizeof(float));
  h_target = (float *)malloc(b * n * sizeof(float));

  // Random number generator
  std::random_device rd;  // Obtain a random number from hardware
  std::mt19937 gen(rd()); // Seed the generator
  std::uniform_real_distribution<> distr(0.f, 1.f); // Define the range

  // Generate and print a random integer
  for (int i = 0; i < b * n; ++i) {
    h_X[i] = distr(gen);
  }

  std::clock_t time_start = std::clock();
  rms_norm_cpu(h_X, h_target, b, n, eps);
  std::clock_t time_end = std::clock();

  double latency_cpu = (double)(time_end - time_start) / CLOCKS_PER_SEC * 1000;
  std::cout << "Latency for rms norm on CPU: " << latency_cpu << std::endl;

  cudaSetDevice(0);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_X, b * n * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, b * n * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_X, h_X, b * n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  size_t grid_dim = b;
  std::function<void(cudaStream_t)> bound_function_rms_norm{std::bind(
      launch_rms_norm, d_X, d_output, b, n, eps, grid_dim, block_dim, stream)};

  float const latency_gpu{measure_performance(bound_function_rms_norm, stream,
                                              num_repeats, num_warmups)};
  std::cout << "Latency for rms norm on GPU, block_dim " << block_dim << ": "
            << latency_gpu << std::endl;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, b * n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  bool success = check_result(h_output, h_target, b * n);
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
