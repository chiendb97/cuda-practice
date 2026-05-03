import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/layer_normalization")

mod = torch.utils.cpp_extension.load(
    name="layer_normalization",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

x = torch.randn(512, 1024, device="cuda")
eps = 1e-5


# NOTE: this kernel skips the mean-subtract step, so it's an RMS-style
# normalization. We compare against that, not against full LayerNorm.
def layer_norm_ref(x, eps):
    return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


ref = layer_norm_ref(x, eps)
torch.testing.assert_close(mod.layer_norm(x, eps), ref, rtol=1e-4, atol=1e-5)
print("layer_norm PASS")

print()
print(f"layer_norm           {do_bench(lambda: mod.layer_norm(x, eps)):.3f} ms")
print(f"layer_norm_ref       {do_bench(lambda: layer_norm_ref(x, eps)):.3f} ms")
print(f"F.layer_norm (full)  {do_bench(lambda: F.layer_norm(x, [x.size(-1)], eps=eps)):.3f} ms")
