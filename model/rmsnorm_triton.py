import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    """One program handles ONE row (one token's full hidden vector)."""
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * n_cols
    out_row_start = out_ptr + row_idx * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols  # BLOCK_SIZE is rounded up to a power of 2, may exceed n_cols

    x = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # mean of squares, computed only over the REAL n_cols values
    # (masked-out slots were loaded as 0.0, which correctly contributes 0 to the sum)
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    x_norm = x * rstd

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * weight

    tl.store(out_row_start + col_offsets, out, mask=mask)


def triton_rmsnorm(x, weight, eps=1e-6):
    """
    x: (..., hidden_size) — any leading shape (batch, seq_len, etc.)
    weight: (hidden_size,)
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
    num_rows, n_cols = x_flat.shape

    out = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (num_rows,)  # launch exactly one program per row

    rmsnorm_kernel[grid](x_flat, weight, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(orig_shape)
