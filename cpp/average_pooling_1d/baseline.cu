#include "../cuda_common.cuh"

__global__
void average_pooling_1d(const float *__restrict__ X, float *Y, int kernel_size, int padding, int stride,
                        int H, int H_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H_out) return;

    float sum = 0;
    int start = max(idx * stride - padding, 0);
    int end   = min(idx * stride + kernel_size - 1 - padding, H - 1);
    for (auto i = start; i <= end; ++i)
        sum += X[i];
    Y[idx] = sum / kernel_size;
}

void launch_average_pooling_1d(const float *d_X, float *d_output, int kernel_size, int padding, int stride,
                                int H, int H_out, int grid_dim, int block_dim, cudaStream_t stream) {
    average_pooling_1d<<<grid_dim, block_dim, 0, stream>>>(d_X, d_output, kernel_size, padding, stride, H, H_out);
}
