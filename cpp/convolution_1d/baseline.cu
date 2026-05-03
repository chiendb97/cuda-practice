#include "../cuda_common.cuh"

__global__
void conv_1d(const float *__restrict__ A, const float *__restrict__ B, float *C, size_t N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float accum = 0;
        for (auto j = -K; j <= K; ++j) {
            if (idx + j >= 0 && idx + j < (int)N)
                accum += A[idx + j] * B[j + K];
        }
        C[idx] = accum;
    }
}

void launch_conv_1d(const float *d_A, const float *d_B, float *d_output, size_t n, int k,
                    int grid_dim, int block_dim, cudaStream_t stream) {
    conv_1d<<<grid_dim, block_dim, 0, stream>>>(d_A, d_B, d_output, n, k / 2);
}
