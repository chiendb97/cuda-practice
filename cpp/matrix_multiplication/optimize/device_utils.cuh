#pragma once

#include <assert.h>

template<typename T>
__device__ __forceinline__ void tiled_mem_cpy(T *src, T *dst, const uint32_t src_stride,
                                              const uint32_t tile_rows, const uint32_t tile_cols) {
    const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * blockDim.y;

    assert(num_threads % tile_cols == 0);

    const auto row_step = num_threads / tile_cols;
    const auto thread_row = thread_idx / tile_cols;
    const auto thread_col = thread_idx % tile_cols;

    for (size_t r = thread_row; r < tile_rows; r += row_step) {
        dst[r * tile_cols + thread_col] = src[r * src_stride + thread_col];
    }
}


template<typename T, uint32_t TILE_ROWS, uint32_t TILE_COLS, uint32_t NUM_THREADS>
__device__ __forceinline__ void tiled_mem_cpy_unrolling(T *src, T *dst, const uint32_t src_stride) {
    const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    constexpr auto row_step = NUM_THREADS / TILE_COLS;
    constexpr auto num_iters = TILE_ROWS / row_step;
    auto thread_row = thread_idx / TILE_COLS;
    const auto thread_col = thread_idx % TILE_COLS;

#pragma unroll
    for (uint32_t i = 0; i < num_iters; ++i) {
        dst[thread_row * TILE_COLS + thread_col] = src[thread_row * src_stride + thread_col];
        thread_row += row_step;
    }
}


template<uint32_t TILE_ROWS, uint32_t TILE_COLS, uint32_t NUM_THREADS, uint32_t VEC_SIZE>
__device__ __forceinline__ void
tiled_mem_cpy_unrolling_vectorized(float4 *src, float4 *dst, const uint32_t src_stride) {
    const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    const uint32_t src_stride_vectorized = src_stride / VEC_SIZE;
    constexpr auto TILE_COLS_VECTORIZED = TILE_COLS / VEC_SIZE;
    constexpr auto row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr auto num_iters = TILE_ROWS / row_step;
    auto thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const auto thread_col = thread_idx % TILE_COLS_VECTORIZED;

#pragma unroll
    for (uint32_t i = 0; i < num_iters; ++i) {
        dst[thread_row * TILE_COLS_VECTORIZED + thread_col] = src[thread_row * src_stride_vectorized + thread_col];
        thread_row += row_step;
    }
}


template<uint32_t TILE_ROWS, uint32_t TILE_COLS, uint32_t NUM_THREADS, uint32_t VEC_SIZE, uint32_t SWIZZLE_BITS>
__device__ __forceinline__ void
tiled_mem_cpy_swizzle(float4 *src, float4 *dst, const uint32_t src_stride) {
    constexpr uint32_t SWIZZLE_MASKS = 0b111 << SWIZZLE_BITS;

    const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    const uint32_t src_stride_vectorized = src_stride / VEC_SIZE;
    constexpr auto TILE_COLS_VECTORIZED = TILE_COLS / VEC_SIZE;
    constexpr auto row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr auto num_iters = TILE_ROWS / row_step;
    auto thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const auto thread_col = thread_idx % TILE_COLS_VECTORIZED;

#pragma unroll
    for (uint32_t i = 0; i < num_iters; ++i) {
        const uint32_t src_index = thread_row * src_stride_vectorized + thread_col;
        uint32_t dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASKS) >> SWIZZLE_BITS);
        dst[dst_index] = src[src_index];
        thread_row += row_step;
    }
}


template<uint32_t TILE_ROWS, uint32_t TILE_COLS, uint32_t NUM_THREADS, uint32_t VEC_SIZE, uint32_t ELEMENTS_PER_THREAD>
__device__ __forceinline__ void
tiled_mem_cpy_load(float4 *src, float4 (&dst_reg)[ELEMENTS_PER_THREAD], const uint32_t src_stride) {
    const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    const uint32_t src_stride_vectorized = src_stride / VEC_SIZE;
    constexpr auto TILE_COLS_VECTORIZED = TILE_COLS / VEC_SIZE;
    constexpr auto row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr auto num_iters = TILE_ROWS / row_step;
    auto thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const auto thread_col = thread_idx % TILE_COLS_VECTORIZED;

#pragma unroll
    for (uint32_t i = 0; i < num_iters; ++i) {
        dst_reg[i] = src[thread_row * src_stride_vectorized + thread_col];
        thread_row += row_step;
    }
}


template<uint32_t TILE_ROWS, uint32_t TILE_COLS, uint32_t NUM_THREADS, uint32_t VEC_SIZE, uint32_t ELEMENTS_PER_THREAD,
    uint32_t SWIZZLE_BITS>
__device__ __forceinline__ void
tiled_mem_cpy_swizzle_store(float4 (&src_reg)[ELEMENTS_PER_THREAD], float4 *dst) {
    constexpr uint32_t SWIZZLE_MASKS = 0b111 << SWIZZLE_BITS;

    const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    constexpr auto TILE_COLS_VECTORIZED = TILE_COLS / VEC_SIZE;
    constexpr auto row_step = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr auto num_iters = TILE_ROWS / row_step;
    auto thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const auto thread_col = thread_idx % TILE_COLS_VECTORIZED;

#pragma unroll
    for (uint32_t i = 0; i < num_iters; ++i) {
        uint32_t dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASKS) >> SWIZZLE_BITS);
        dst[dst_index] = src_reg[i];
        thread_row += row_step;
    }
}


__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}


// the stmatrix ptx instruction works for sm_90 and above
// this is a workaround
// this is innefficient, access pattern results in bad coalescing
__device__ __forceinline__ void stmatrix_m16n8(
    half *dst,
    half (&reg)[4],
    unsigned int dst_stride_bytes
) {
    const unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_)[2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst);
    dst_stride_bytes /= sizeof(uint32_t);
    unsigned int fragment_row = laneIdx / 4;
    const unsigned int fragment_col = laneIdx % 4;

    // 4 adjacent threads storing 4 bytes each == 16 byte transactions
    dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[0];
    fragment_row += 8;
    dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[1];
}


// loads an MMA tile directly from global memory
// this is innefficient, access pattern results in bad coalescing
__device__ __forceinline__ void ldmatrix_m16n8_gmem(
    half *src,
    half (&reg)[4],
    unsigned int src_stride_bytes
) {
    const unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_)[2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src);
    src_stride_bytes /= sizeof(uint32_t);
    unsigned int fragment_row = laneIdx / 4;
    const unsigned int fragment_col = laneIdx % 4;

    // 4 adjacent threads storing 4 bytes each == 16 byte transactions
    reg_[0] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
    fragment_row += 8;
    reg_[1] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
}


constexpr unsigned int int_log2(unsigned int x) {
    unsigned int result = 0;
    while (x >>= 1) {
        result++;
    }
    return result;
}
