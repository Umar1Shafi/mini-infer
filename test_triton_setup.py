import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each "program" (parallel instance) handles one BLOCK_SIZE-sized chunk
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # guard against reading/writing past the array's end

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x, y):
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # how many parallel "programs" to launch
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---- Test ----
torch.manual_seed(0)
x = torch.randn(100_000, device="cuda")
y = torch.randn(100_000, device="cuda")

triton_result = triton_add(x, y)
pytorch_result = x + y

max_diff = (triton_result - pytorch_result).abs().max().item()
print(f"Max difference: {max_diff}")
print("Match!" if max_diff == 0.0 else "MISMATCH")
