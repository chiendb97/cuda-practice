#include "../cuda_common.cuh"

__global__
void vector_add(const float *d_input1, const float *d_input2, float *d_output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_output[idx] = d_input1[idx] + d_input2[idx];
    }
}

void launch_vector_add(const float *d_input1, const float *d_input2, float *d_output, size_t n,
                       int grid_dim, int block_dim, cudaStream_t stream) {
    vector_add<<<grid_dim, block_dim, 0, stream>>>(d_input1, d_input2, d_output, n);
}
