#include "../cuda_common.cuh"
#include <cuda_fp16.h>
#include <mma.h>

template <typename T, typename ACCUM, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void matrix_multiplication_kernel(int M, int N, int K, const T *A,
                                             int lda, const T *B, int ldb,
                                             ACCUM *C, int ldc, float alpha,
                                             float beta) {
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T,
                           WMMA_FRAG_LAYOUT_A> a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T,
                           WMMA_FRAG_LAYOUT_B> b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, ACCUM> acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, ACCUM> c_frag{};

    nvcuda::wmma::fill_fragment(acc_frag, static_cast<ACCUM>(0));

    for (unsigned k = 0; k < K; k += WMMA_K) {
        T const *matrix_mma_a_mptr{A + warpM * WMMA_M + k * lda};
        T const *matrix_mma_b_mptr{B + warpN * WMMA_N + k * ldb};
        nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
        nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    ACCUM *matrix_mma_c_mptr{C + warpM * WMMA_M + warpN * WMMA_N * ldc};
    nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc, nvcuda::wmma::mem_col_major);
    for (uint32_t i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc, nvcuda::wmma::mem_col_major);
}

void launch_matmul_tensor_core(int M, int N, int K, const half *A, const half *B,
                               float *C, cudaStream_t stream) {
    constexpr int WMMA_M{16}, WMMA_N{16}, WMMA_K{16};
    constexpr int num_warps_x = 4, num_warps_y = 4;

    dim3 block_dim(num_warps_x * 32, num_warps_y);
    dim3 grid_dim((M + WMMA_M * num_warps_x - 1) / (WMMA_M * num_warps_x),
                  (N + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y));

    matrix_multiplication_kernel<half, float, WMMA_M, WMMA_N, WMMA_K,
                                 nvcuda::wmma::col_major, nvcuda::wmma::row_major>
        <<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, M, B, N, C, M, 1.f, 0.f);
}
