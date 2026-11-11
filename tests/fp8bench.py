#!/usr/bin/env python3
"""Time SGLang's w8a8 block-FP8 matmul for one shape with whatever config the runtime picks
(default if no tuned JSON is present in the configs dir, tuned if it is). Run twice to compare."""
import sys, torch
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton as w8a8_block_fp8_matmul
N, K = int(sys.argv[1]), int(sys.argv[2]); bn = bk = 32
torch.manual_seed(0); dev = "cuda"
B = (torch.rand(N, K, device=dev) - 0.5).to(torch.float8_e4m3fn)
Bs = torch.rand((N + bn - 1) // bn, (K + bk - 1) // bk, device=dev) * 1e-2
out = []
for M in (1, 4, 16, 64, 128, 512, 2048):
    A = (torch.rand(M, K, device=dev) - 0.5).to(torch.float8_e4m3fn)
    As = torch.rand(M, (K + bk - 1) // bk, device=dev) * 1e-2
    f = lambda: w8a8_block_fp8_matmul(A, B, As, Bs, [bn, bk], torch.bfloat16)
    for _ in range(10): f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): f()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); [g.replay() for _ in range(10)]; e.record(); torch.cuda.synchronize()
    out.append(f"M={M}:{s.elapsed_time(e) / 200 * 1000:.1f}us")
print(" ".join(out))
