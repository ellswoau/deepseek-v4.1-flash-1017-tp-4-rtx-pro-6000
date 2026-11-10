/* Minimal STREAM-style host memory bandwidth test (Copy/Scale/Add/Triad), OpenMP.
 * Arrays are 4 GiB each (far beyond LLC). Reports best-of-N GB/s (1 GB = 1e9 bytes),
 * counting bytes the way STREAM does (no write-allocate traffic).
 * Build: gcc -O3 -march=native -fopenmp stream.c -o stream
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N (512UL * 1024 * 1024) /* doubles per array = 4 GiB */
#define NTIMES 10

int main(void) {
    double *a = aligned_alloc(64, N * sizeof(double));
    double *b = aligned_alloc(64, N * sizeof(double));
    double *c = aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; j++) { a[j] = 1.0; b[j] = 2.0; c[j] = 0.0; }

    const char *name[4] = {"Copy", "Scale", "Add", "Triad"};
    const double bytes[4] = {2.0 * N * 8, 2.0 * N * 8, 3.0 * N * 8, 3.0 * N * 8};
    double best[4] = {1e30, 1e30, 1e30, 1e30}, s = 3.0;
    for (int k = 0; k < NTIMES; k++) {
        double t;
        t = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < N; j++) c[j] = a[j];
        t = omp_get_wtime() - t; if (k && t < best[0]) best[0] = t;
        t = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < N; j++) b[j] = s * c[j];
        t = omp_get_wtime() - t; if (k && t < best[1]) best[1] = t;
        t = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < N; j++) c[j] = a[j] + b[j];
        t = omp_get_wtime() - t; if (k && t < best[2]) best[2] = t;
        t = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < N; j++) a[j] = b[j] + s * c[j];
        t = omp_get_wtime() - t; if (k && t < best[3]) best[3] = t;
    }
    printf("threads=%d array=%.1f GiB\n", omp_get_max_threads(), N * 8.0 / (1 << 30));
    for (int i = 0; i < 4; i++) printf("%-6s %8.1f GB/s\n", name[i], bytes[i] / best[i] / 1e9);
    return 0;
}
