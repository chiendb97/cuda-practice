import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/vector_addition")

mod = torch.utils.cpp_extension.load(
    name="vector_addition",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/vectorized_memory_access.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

n = 1 << 22
a = torch.randn(n, device="cuda")
b = torch.randn(n, device="cuda")
ref = a + b

torch.testing.assert_close(mod.vector_add(a, b), ref)
print("vector_add PASS")
torch.testing.assert_close(mod.vector_add_vec4(a, b), ref)
print("vector_add_vec4 PASS")

print()
print(f"vector_add      {do_bench(lambda: mod.vector_add(a, b)):.3f} ms")
print(f"vector_add_vec4 {do_bench(lambda: mod.vector_add_vec4(a, b)):.3f} ms")
print(f"torch (a + b)   {do_bench(lambda: a + b):.3f} ms")
