import torch, time
from model.rmsnorm import RMSNorm
from model.rmsnorm_triton import triton_rmsnorm

hidden_size = 896
norm = RMSNorm(hidden_size).to("cuda")
x = torch.randn(1, 2000, hidden_size, device="cuda")  # a long-ish sequence, to make timing meaningful

def timed(fn, iters=200):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000  # ms per call

# Warm-up (first call compiles the Triton kernel; don't time that)
_ = norm(x)
_ = triton_rmsnorm(x, norm.weight, eps=norm.eps)

pytorch_ms = timed(lambda: norm(x))
triton_ms = timed(lambda: triton_rmsnorm(x, norm.weight, eps=norm.eps))

print(f"PyTorch RMSNorm: {pytorch_ms:.4f} ms/call")
print(f"Triton RMSNorm:  {triton_ms:.4f} ms/call")
print(f"Triton is {pytorch_ms / triton_ms:.2f}x {'faster' if triton_ms < pytorch_ms else 'slower'}")
