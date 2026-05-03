#include <cub/device/device_scan.cuh>
#include "../cuda_common.cuh"

void launch_prefix_sum_cub(const float *d_X, float *d_output, int n,
                           int grid_dim, int block_dim, cudaStream_t stream) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_X, d_output, n, stream);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_X, d_output, n, stream);
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}
