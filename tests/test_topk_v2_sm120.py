"""Kernel check on SM120: topk_transform_paged_v2 (+plan) and amax_topk_blocks vs torch reference, at 1M-context width."""
import torch, time
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2
from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax_topk_blocks
torch.manual_seed(0); dev = "cuda"
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
W = 262144  # 1M context / ratio 4
for bs, K in [(1, 512), (8, 512), (16, 512), (96, 512), (16, 2048)]:
    lens = torch.randint(1, 60000, (bs,), device=dev, dtype=torch.int32); lens[0] = 7500; lens[-1] = 300  # one row <= K
    scores = torch.randn(bs, W, device=dev, dtype=torch.float32)
    plan = plan_topk_v2(lens)
    out = torch.empty(bs, K, dtype=torch.int32, device=dev)
    topk_transform_paged_v2(scores, lens, None, out, 1, plan); torch.cuda.synchronize()
    bad = 0
    for b in range(bs):
        L = int(lens[b]); k = min(K, L)
        ref = set(torch.topk(scores[b, :L], k).indices.tolist())
        got = set(x for x in out[b].tolist() if x >= 0)
        if got != ref: bad += 1
    # timing vs torch full-width (what SM120 does today)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): topk_transform_paged_v2(scores, lens, None, out, 1, plan)
    torch.cuda.synchronize(); tv2 = (time.perf_counter() - t) / 20 * 1e3
    m = torch.arange(W, device=dev)[None] >= lens[:, None]
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): scores.clone().masked_fill_(m, -float("inf")).topk(K, dim=1, sorted=False)
    torch.cuda.synchronize(); tt = (time.perf_counter() - t) / 20 * 1e3
    print(f"topk_v2 bs={bs:3d} K={K}: rows mismatched {bad}/{bs}   v2 {tv2:.3f} ms vs torch full-width {tt:.3f} ms")
# amax_topk_blocks (candidate source, block 8, 2048 blocks)
bs = 16; lens = torch.randint(20000, 60000, (bs,), device=dev, dtype=torch.int32)
scores = torch.randn(bs, W, device=dev, dtype=torch.float32)
nb = (lens + 7) // 8
blocks = amax_topk_blocks(scores, lens, nb, 2048); torch.cuda.synchronize(); bad = 0
for b in range(bs):
    L = int(lens[b]); s = scores[b, :L]; pad = (-L) % 8
    bm = torch.nn.functional.pad(s, (0, pad), value=-float("inf")).view(-1, 8).amax(1); bm[-1] = float("inf")
    ref = set(torch.topk(bm, min(2048, bm.numel())).indices.tolist()); got = set(x for x in blocks[b].tolist() if x >= 0)
    bad += got != ref
print(f"amax_topk_blocks bs={bs}: rows mismatched {bad}/{bs}")
print("KERNEL_TEST_DONE")
