import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/parallel_reduction")

mod = torch.utils.cpp_extension.load(
    name="parallel_reduction",
    sources=[
        f"{SRC}/neighbored_pair.cu",
        f"{SRC}/interleaved_pair.cu",
        f"{SRC}/neighbored_pair_less.cu",
        f"{SRC}/loop_unrolling.cu",
        f"{SRC}/warp_unrolling.cu",
        f"{SRC}/complete_unrolling.cu",
        f"{SRC}/template_function.cu",
        f"{SRC}/template_function_shared_memory.cu",
        f"{SRC}/warp_shuffle_instruction.cu",
        f"{SRC}/bindings.cpp",
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

n = 1 << 22
x = torch.randint(0, 10, (n,), dtype=torch.int32, device="cuda")
ref = x.sum()

variants = [
    "reduce_neighbored_pair",
    "reduce_interleaved_pair",
    "reduce_neighbored_pair_less",
    "reduce_loop_unrolling",
    "reduce_warp_unrolling",
    "reduce_complete_unrolling",
    "reduce_template_function",
    "reduce_template_function_shared_memory",
    "reduce_warp_shuffle_instruction",
]

for name in variants:
    fn = getattr(mod, name)
    torch.testing.assert_close(fn(x), ref)
    print(f"{name} PASS")

print()
for name in variants:
    fn = getattr(mod, name)
    print(f"{name:42s} {do_bench(lambda: fn(x)):.3f} ms")
print(f"{'torch x.sum()':42s} {do_bench(lambda: x.sum()):.3f} ms")
