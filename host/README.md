# Host preparation

## Transparent huge pages for shared memory (required for full speed)

SGLang's native Engram host table (`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared`) keeps the two Engram tables in one memfd shared mapping, about 189 GiB in total. The GPUs read rows from it directly over PCIe. Without huge pages, every row lookup costs a TLB miss. SGLang warns at startup:

```
engram host table layer 1: layout=shared, ... 0 MiB in huge pages (0%). No huge pages: expect ~10x slower lookups
(one TLB miss per row); transparent_hugepage/shmem_enabled is 'never'
```

Many distributions default `shmem_enabled` to `never`. Set it to `advise`, which only affects regions that request huge pages:

```bash
sudo install -m 644 host/ds41-thp-shmem.conf /etc/tmpfiles.d/
sudo systemd-tmpfiles --create /etc/tmpfiles.d/ds41-thp-shmem.conf   # applies now; tmpfiles re-applies at boot
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled                # ... [advise] ...
```

Restart the engine afterwards, because existing mappings are not converted. A good start logs `... 96681 MiB resident, 96680 MiB in huge pages (100%)` for each layer, and `ShmemHugePages` in `/proc/meminfo` shows about 189 GiB.

Measured effect (agent-shaped, C=8): **599 → 699 tok/s**, with median TTFT 0.69 → 0.48 s.

## Other host settings used

- Kernel command line: `intel_iommu=on iommu=pt`. `vm.swappiness=10`.
- `nvidia-persistenced` running, persistence mode on.
- Docker: `memlock` unlimited and `IPC_LOCK` (both in compose.yaml). The locked Engram tables and HiCache host pool need them.
- Everything large (checkpoint, Docker data root, kernel caches) on local NVMe.
