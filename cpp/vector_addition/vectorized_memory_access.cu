#include "../cuda_common.cuh"

__global__ void vector_add_vec4(const float4 *d_input1, const float4 *d_input2,
                                float4 *d_output, size_t n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float4 a = d_input1[idx];
        const float4 b = d_input2[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        d_output[idx] = c;
    }
}

void launch_vector_add_vec4(const float *d_input1, const float *d_input2,
                            float *d_output, size_t n, int grid_dim, int block_dim,
                            cudaStream_t stream) {
    vector_add_vec4<<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const float4 *>(d_input1),
        reinterpret_cast<const float4 *>(d_input2),
        reinterpret_cast<float4 *>(d_output), n / 4);
}
