#include <cub/device/device_scan.cuh>
#include "../cuda_common.cuh"

struct CustomMultiple {
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a, const T &b) const { return a * b; }
};

void launch_prefix_multiple(const float *d_X, float *d_output, int n,
                            int grid_dim, int block_dim, cudaStream_t stream) {
    CustomMultiple op;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_X, d_output, op, n, stream);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_X, d_output, op, n, stream);
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}
