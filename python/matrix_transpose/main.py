import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/matrix_transpose")

mod = torch.utils.cpp_extension.load(
    name="matrix_transpose",
    sources=[f"{SRC}/naive.cu", f"{SRC}/shared_memory.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

x = torch.randint(0, 1 << 16, (1024, 1024), dtype=torch.int32, device="cuda")

# Copy variants: output should equal input
torch.testing.assert_close(mod.copy_row(x), x)
print("copy_row PASS")
torch.testing.assert_close(mod.copy_column(x), x)
print("copy_column PASS")

# Transpose variants
ref = x.t().contiguous()
for name in ["transpose_row", "transpose_column",
             "transpose_diagonal_row", "transpose_diagonal_column",
             "transpose_shared_memory", "transpose_shared_memory_padding"]:
    torch.testing.assert_close(getattr(mod, name)(x), ref)
    print(f"{name} PASS")

print()
for name in ["copy_row", "copy_column",
             "transpose_row", "transpose_column",
             "transpose_diagonal_row", "transpose_diagonal_column",
             "transpose_shared_memory", "transpose_shared_memory_padding"]:
    fn = getattr(mod, name)
    print(f"{name:34s} {do_bench(lambda: fn(x)):.3f} ms")
print(f"{'torch x.t().contiguous()':34s} {do_bench(lambda: x.t().contiguous()):.3f} ms")
