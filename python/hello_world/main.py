import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/hello_world")

mod = torch.utils.cpp_extension.load(
    name="hello_world",
    sources=[f"{SRC}/hello_cuda.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

mod.hello(16, 16, 4, 4)
torch.cuda.synchronize()

ms = do_bench(lambda: mod.hello(16, 16, 4, 4))
print(f"hello {ms:.3f} ms")
