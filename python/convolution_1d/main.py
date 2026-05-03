import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/convolution_1d")

mod = torch.utils.cpp_extension.load(
    name="convolution_1d",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

n, k = 1 << 20, 7
x = torch.randn(n, device="cuda")
w = torch.randn(k, device="cuda")


def conv_1d_ref(x, w):
    # The CUDA kernel does correlation (no kernel flip), so use F.conv1d as-is.
    k = w.size(0)
    return F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1), padding=k // 2).view(-1)


ref = conv_1d_ref(x, w)
torch.testing.assert_close(mod.conv_1d(x, w), ref, rtol=1e-4, atol=1e-4)
print("conv_1d PASS")

print()
print(f"conv_1d     {do_bench(lambda: mod.conv_1d(x, w)):.3f} ms")
print(f"torch (ref) {do_bench(lambda: conv_1d_ref(x, w)):.3f} ms")
