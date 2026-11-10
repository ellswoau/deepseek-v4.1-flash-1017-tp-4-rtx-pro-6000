# DeepSeek-V4.1-Flash on 4 x RTX PRO 6000 (SM120): stock SGLang v0.5.23 + SM120 runtime fixes.
# Engram tables use SGLang's native host table (locked, huge-page shared memory); HiCache is on.
FROM lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f
# lmsysorg/sglang:v0.5.23-cu130 (2026-10-30)

# Fix 1: extra-source 64-token page split for the SM120 FlashInfer sparse-MLA prefill (Apache-2.0, see NOTICE)
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
# Fixes 2 + 3: DSV4 indexer metadata + candidate-indexer guard, applied at import time (MIT)
COPY adapter/sitecustomize.py /opt/ds41/adapter/sitecustomize.py
COPY launch.sh /opt/ds41/launch.sh
ENV PYTHONPATH=/opt/ds41/adapter DS41_SM120_INDEXER_FIX=1
EXPOSE 8010
ENTRYPOINT ["bash", "/opt/ds41/launch.sh"]
