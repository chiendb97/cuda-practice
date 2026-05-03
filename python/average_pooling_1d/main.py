import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/average_pooling_1d")

mod = torch.utils.cpp_extension.load(
    name="average_pooling_1d",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

H, kernel_size, stride, padding = 1 << 20, 3, 1, 1
x = torch.randn(H, device="cuda")


def avg_pool_ref(x, kernel_size, stride, padding):
    return F.avg_pool1d(x.view(1, 1, -1), kernel_size, stride, padding).view(-1)


ref = avg_pool_ref(x, kernel_size, stride, padding)
torch.testing.assert_close(mod.avg_pool_1d(x, kernel_size, padding, stride), ref, rtol=1e-4, atol=1e-5)
print("avg_pool_1d PASS")

print()
print(f"avg_pool_1d {do_bench(lambda: mod.avg_pool_1d(x, kernel_size, padding, stride)):.3f} ms")
print(f"torch (ref) {do_bench(lambda: avg_pool_ref(x, kernel_size, stride, padding)):.3f} ms")
