#pragma once

#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ void warp_group_arrive()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wgmma_wait_group()
{
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int N>
__device__ __forceinline__ void set_max_reg_dec()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(N));
}

template <int N>
__device__ __forceinline__ void set_max_reg_inc()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(N));
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint32_t x)
{
    return (x & 0x3FFFF) >> 4;
}

template <uint32_t smem_width_bytes = 128>
__device__ uint64_t make_smem_desc(half *ptr)
{
    constexpr uint64_t leading_dim_byte_offset = 16;
    constexpr uint64_t stride_dim_byte_offset = smem_width_bytes;
    constexpr uint64_t swizzle_128b = 1ull;

    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    return matrix_descriptor_encode(addr) |
           (matrix_descriptor_encode(leading_dim_byte_offset) << 16) |
           (matrix_descriptor_encode(stride_dim_byte_offset * 8) << 32) |
           (swizzle_128b << 62);
}

template <uint32_t smem_width_bytes = 128>
__device__ __forceinline__ void wgmma_m64n256k16_f32_f16_f16(float D[128], half *A_block_smem, half *B_block_smem)
{
    uint64_t A_desc = make_smem_desc<smem_width_bytes>(A_block_smem);
    uint64_t B_desc = make_smem_desc<smem_width_bytes>(B_block_smem);

    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,"
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,"
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, " // D (accumulator registers)

        "%128, %129, " // A_desc, B_desc (shared memory descriptors)

        "1, 1, 1, 0, 0;" //  scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b
        // scale-d=1: compute D = A * B + D rather than D = A * B
        // imm-scale-a=1: compute A = A * 1.0f (no scaling, A can optionally be negated if you pass -1)
        // imm-scale-b=1: compute B = B * 1.0f (no scaling, B can optionally be negated if you pass -1)
        // imm-trans-a=0: A is k-major in shared memory
        // imm-trans-b=0: B is k-major in shared memory
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]),
          "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]),
          "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]),
          "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63]),
          "+f"(D[64]), "+f"(D[65]), "+f"(D[66]), "+f"(D[67]), "+f"(D[68]), "+f"(D[69]), "+f"(D[70]), "+f"(D[71]), "+f"(D[72]), "+f"(D[73]), "+f"(D[74]), "+f"(D[75]), "+f"(D[76]), "+f"(D[77]), "+f"(D[78]), "+f"(D[79]),
          "+f"(D[80]), "+f"(D[81]), "+f"(D[82]), "+f"(D[83]), "+f"(D[84]), "+f"(D[85]), "+f"(D[86]), "+f"(D[87]), "+f"(D[88]), "+f"(D[89]), "+f"(D[90]), "+f"(D[91]), "+f"(D[92]), "+f"(D[93]), "+f"(D[94]), "+f"(D[95]),
          "+f"(D[96]), "+f"(D[97]), "+f"(D[98]), "+f"(D[99]), "+f"(D[100]), "+f"(D[101]), "+f"(D[102]), "+f"(D[103]), "+f"(D[104]), "+f"(D[105]), "+f"(D[106]), "+f"(D[107]), "+f"(D[108]), "+f"(D[109]), "+f"(D[110]), "+f"(D[111]),
          "+f"(D[112]), "+f"(D[113]), "+f"(D[114]), "+f"(D[115]), "+f"(D[116]), "+f"(D[117]), "+f"(D[118]), "+f"(D[119]), "+f"(D[120]), "+f"(D[121]), "+f"(D[122]), "+f"(D[123]), "+f"(D[124]), "+f"(D[125]), "+f"(D[126]), "+f"(D[127])
        : "l"(A_desc), "l"(B_desc));
}