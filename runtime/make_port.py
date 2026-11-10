#!/usr/bin/env python3
"""Port Sybil's SM120 sparse-prefill fix onto SGLang v0.5.23's flash_mla_sm120.py.

Only Sybil's functional hunks are applied; v0.5.23's own additions (decode capability probe,
GLM triton option) are kept. Every replacement must match exactly once.
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()

def once(old, new, what):
    global text
    n = text.count(old)
    assert n == 1, f"{what}: expected 1 match, found {n}"
    text = text.replace(old, new)
    print("applied:", what)

# 1. Split the V4.1 ratio-1/2 "extra" source (256/128-token pages) to 64-token pages for the
#    FlashInfer SM120 prefill kernel, with its own scratch buffer so it cannot overwrite SWA.
once(
    "    output = q2.new_empty((num_tokens, num_heads, head_dim_v), dtype=torch.bfloat16)\n"
    "    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=dev)\n"
    "    _sparse_mla_sm120_paged_attention(",
    "    # [DS41 port of Sybil SM120 fix] V4.1 ratio-1/2 sources use 256/128-token pages. The prefill\n"
    "    # specialization needs 64-token pages for both sources, with independent\n"
    "    # scratch buffers so converting the extra source cannot overwrite SWA.\n"
    "    if extra_kv_u8 is not None and extra_kv_u8.shape[1] != _PBS_DST:\n"
    "        extra_kv_u8 = _split_kv_pages_to_64(\n"
    "            extra_kv_u8, extra_kv_u8.shape[1],\n"
    "            touched_indices=extra_idx, buffer_tag=\"extra\",\n"
    "        )\n"
    "    output = q2.new_empty((num_tokens, num_heads, head_dim_v), dtype=torch.bfloat16)\n"
    "    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=dev)\n"
    "    _sparse_mla_sm120_paged_attention(",
    "extra-source 64-page split in prefill")

# 2. buffer_tag parameter on the splitter
once(
    "    touched_indices: Optional[torch.Tensor] = None,\n) -> torch.Tensor:\n    \"\"\"Split pbs=N footer-format pages",
    "    touched_indices: Optional[torch.Tensor] = None,\n    buffer_tag: str = \"primary\",\n) -> torch.Tensor:\n    \"\"\"Split pbs=N footer-format pages",
    "buffer_tag parameter")

# 3./4. independent scratch + mask buffers per tag
once('    key = f"flash_mla_sm120_split:{dev}"\n', '    key = f"flash_mla_sm120_split:{dev}:{buffer_tag}"\n', "split buffer keyed by tag")
once('        mkey = f"flash_mla_sm120_mask:{dev}"\n', '        mkey = f"flash_mla_sm120_mask:{dev}:{buffer_tag}"\n', "mask buffer keyed by tag")

# sanity: the names the new code relies on must exist in this file
for name in ("extra_kv_u8", "extra_idx", "_PBS_DST", "def _split_kv_pages_to_64"):
    assert name in text, f"missing symbol {name}"
dst.write_text(text)
print("wrote", dst)
