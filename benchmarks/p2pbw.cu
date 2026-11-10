// Raw P2P copy bandwidth: single pair (uni + bidir) and all 12 directed pairs at once.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)
const size_t SZ = 256ull << 20; const int IT = 20;
int n; void* buf[8][2];
double run(std::vector<std::pair<int,int>> pr, bool ce_dst) {
  std::vector<cudaStream_t> st(pr.size()); cudaEvent_t a, b;
  cudaSetDevice(0); cudaEventCreate(&a); cudaEventCreate(&b);
  for (size_t i = 0; i < pr.size(); i++) { cudaSetDevice(ce_dst ? pr[i].second : pr[i].first); cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking); }
  for (int d = 0; d < n; d++) { cudaSetDevice(d); cudaDeviceSynchronize(); }
  cudaSetDevice(0); cudaEventRecord(a, 0);
  for (size_t i = 0; i < pr.size(); i++) cudaStreamWaitEvent(st[i], a, 0);
  for (int k = 0; k < IT; k++) for (size_t i = 0; i < pr.size(); i++)
    cudaMemcpyPeerAsync(buf[pr[i].second][1], pr[i].second, buf[pr[i].first][0], pr[i].first, SZ, st[i]);
  for (int d = 0; d < n; d++) { cudaSetDevice(d); cudaDeviceSynchronize(); }
  cudaSetDevice(0); cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  for (auto& s : st) cudaStreamDestroy(s);
  return (double)SZ * IT * pr.size() / (ms / 1e3) / 1e9;
}
int main() {
  CK(cudaGetDeviceCount(&n)); printf("devices: %d\n", n);
  for (int i = 0; i < n; i++) { cudaSetDevice(i); for (int j = 0; j < n; j++) if (i != j) { int ok; cudaDeviceCanAccessPeer(&ok, i, j); if (ok) cudaDeviceEnablePeerAccess(j, 0); }
    CK(cudaMalloc(&buf[i][0], SZ)); CK(cudaMalloc(&buf[i][1], SZ)); }
  printf("\nUnidirectional GB/s (row=src, col=dst), copy engine on src:\n     ");
  for (int j = 0; j < n; j++) printf("%7d", j); printf("\n");
  for (int i = 0; i < n; i++) { printf("%4d ", i); for (int j = 0; j < n; j++) printf("%7.1f", i == j ? 0.0 : run({{i,j}}, false)); printf("\n"); }
  printf("\nBidirectional pair GB/s (sum of both directions):\n");
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) printf("  %d<->%d %6.1f\n", i, j, run({{i,j},{j,i}}, false));
  std::vector<std::pair<int,int>> all; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) all.push_back({i,j});
  double t = run(all, false), t2 = run(all, true);
  printf("\nAll %zu directed pairs at once (all-to-all pattern):\n  total %.1f GB/s, per GPU egress %.1f GB/s  (src copy engines)\n  total %.1f GB/s, per GPU egress %.1f GB/s  (dst copy engines)\n", all.size(), t, t / n, t2, t2 / n);
  return 0;
}
