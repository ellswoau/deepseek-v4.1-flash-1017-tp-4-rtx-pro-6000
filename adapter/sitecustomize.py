"""DS41 SM120 runtime fixes for SGLang v0.5.23 (enabled only when DS41_SM120_INDEXER_FIX=1).

1. Indexer metadata (ported from Sybil's adapter/sitecustomize.py): V4.1 ratio-1/2 indexers
   always call the FP4 DeepGEMM kernel. SM120 needs DeepGEMM's split planner metadata even when
   the FP8 indexer would take the torch/JIT path; upstream never sets force_deep_gemm_metadata.

2. Candidate indexer guard (new, for v0.5.23; cf. sglang#44445): make_candidate_indexer enables
   the sparse-table candidate path whenever is_sm100_or_newer() (capability >= 10, so SM120
   too), but DeepGEMM's paged *sparse* MQA logits are only enabled on SM100
   (DEEPGEMM_BLACKWELL), so startup raises on SM120. When that kernel is unavailable we take
   the dense-blocks path the function already uses for candidate_topk_blocks <= 0. Unlike the
   "is_sm100_or_newer() == 10" patch from #44445, this keeps DeepGEMM's dense FP4 indexer
   (prefill/decode) on SM120.

Sybil's Engram row-store hook is NOT used: v0.5.23 has a native Engram host table
(SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1), and the method Sybil replaced no longer exists.
"""
import functools
import importlib.abc
import importlib.machinery
import logging
import os
import sys

METADATA = "sglang.srt.layers.attention.dsv4.metadata"
V41_INDEXER = "sglang.srt.layers.attention.dsv4.v41_indexer"
log = logging.getLogger("ds41.sm120")


def _patch_metadata(module):
    cls = module.PagedIndexerMetadata
    original = cls.__post_init__

    def post_init(self):
        if module._IS_SM120 and self.compress_ratio in (1, 2):
            self.force_deep_gemm_metadata = True
        original(self)

    cls.__post_init__ = post_init


def _patch_v41_indexer(module):
    original = module.make_candidate_indexer

    @functools.wraps(original)
    def make_candidate_indexer(**kwargs):
        from sglang.srt.layers.deep_gemm_wrapper.configurer import (
            DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
        )

        if DEEPGEMM_PAGED_SPARSE_MQA_LOGITS or kwargs.get("candidate_topk_blocks", 0) <= 0:
            return original(**kwargs)
        # Same objects the original builds before its early "return dense_blocks, dense_blocks";
        # every argument unchanged (candidate_topk_blocks included).
        dense_blocks_mod = importlib.import_module(module.__name__ + ".dense_blocks")
        use_deep_gemm_prefill = module._use_deep_gemm_prefill()
        dense_blocks = dense_blocks_mod.DenseBlocksBackend(
            token_to_kv_pool=kwargs["token_to_kv_pool"],
            req_to_token=kwargs["req_to_token"],
            candidate_topk_blocks=kwargs["candidate_topk_blocks"],
            candidate_block_size=kwargs["candidate_block_size"],
            use_deep_gemm_prefill=use_deep_gemm_prefill,
        )
        log.warning(
            "DS41 SM120: DeepGEMM paged sparse MQA logits unavailable; using dense-blocks "
            "candidate indexer (use_deep_gemm_prefill=%s)", use_deep_gemm_prefill,
        )
        return dense_blocks, dense_blocks

    module.make_candidate_indexer = make_candidate_indexer


PATCHES = {METADATA: _patch_metadata, V41_INDEXER: _patch_v41_indexer}


class _Loader(importlib.abc.Loader):
    def __init__(self, original, patch):
        self.original, self.patch = original, patch

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        patch = PATCHES.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader, patch)
        return spec


if os.environ.get("DS41_SM120_INDEXER_FIX") == "1":
    sys.meta_path.insert(0, _Finder())
