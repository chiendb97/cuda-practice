import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/rms_normalization")

mod = torch.utils.cpp_extension.load(
    name="rms_normalization",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/warp_shuffle_instruction.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def rms_norm_ref(x, eps):
    return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


x = torch.randn(512, 1024, device="cuda")
eps = 1e-6
ref = rms_norm_ref(x, eps)

torch.testing.assert_close(mod.rms_norm(x, eps), ref, rtol=1e-4, atol=1e-5)
print("rms_norm PASS")
torch.testing.assert_close(mod.rms_norm_warp(x, eps), ref, rtol=1e-4, atol=1e-5)
print("rms_norm_warp PASS")

print()
print(f"rms_norm      {do_bench(lambda: mod.rms_norm(x, eps)):.3f} ms")
print(f"rms_norm_warp {do_bench(lambda: mod.rms_norm_warp(x, eps)):.3f} ms")
print(f"torch (ref)   {do_bench(lambda: rms_norm_ref(x, eps)):.3f} ms")
