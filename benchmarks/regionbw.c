/* Per-region host memory bandwidth: allocate 1 GiB hugetlb pages, read their physical addresses from
 * /proc/self/pagemap (root), split them at a physical boundary (default 0x8080000000 = start of the
 * 4-way interleaved range on this host), and run STREAM Copy/Triad over pages from each region only.
 * Build: gcc -O3 -march=native -fopenmp regionbw.c -o regionbw      Run as root: ./regionbw <npages> [boundary_hex]
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << 26)
#endif
#define GB (1UL << 30)
#define PER_ARRAY 8 /* GiB per array -> 3 arrays = 24 pages per region */

static uint64_t phys_of(void *va) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    uint64_t e = 0;
    pread(fd, &e, 8, ((uintptr_t)va / 4096) * 8);
    close(fd);
    return (e & ((1ULL << 55) - 1)) * 4096;
}

static void run(const char *name, double **p) { /* p[0..23]: a = p[0..7], b = p[8..15], c = p[16..23] */
    const size_t n = GB / sizeof(double);
    double **a = p, **b = p + PER_ARRAY, **c = p + 2 * PER_ARRAY, s = 3.0;
    for (int k = 0; k < 3 * PER_ARRAY; k++) {
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) p[k][j] = 1.0 + k;
    }
    double best_copy = 1e30, best_triad = 1e30;
    for (int it = 0; it < 6; it++) {
        double t = omp_get_wtime();
        for (int k = 0; k < PER_ARRAY; k++) {
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < n; j++) c[k][j] = a[k][j];
        }
        t = omp_get_wtime() - t; if (it && t < best_copy) best_copy = t;
        t = omp_get_wtime();
        for (int k = 0; k < PER_ARRAY; k++) {
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < n; j++) a[k][j] = b[k][j] + s * c[k][j];
        }
        t = omp_get_wtime() - t; if (it && t < best_triad) best_triad = t;
    }
    double bytes = (double)PER_ARRAY * GB;
    printf("%-34s Copy %7.1f GB/s   Triad %7.1f GB/s   (threads=%d)\n", name,
           2 * bytes / best_copy / 1e9, 3 * bytes / best_triad / 1e9, omp_get_max_threads());
}

int main(int argc, char **argv) {
    int npages = argc > 1 ? atoi(argv[1]) : 600;
    uint64_t boundary = argc > 2 ? strtoull(argv[2], 0, 16) : 0x8080000000ULL;
    double **lo = calloc(npages, sizeof *lo), **hi = calloc(npages, sizeof *hi);
    int nlo = 0, nhi = 0;
    for (int i = 0; i < npages; i++) {
        void *v = mmap(0, GB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (v == MAP_FAILED) { printf("got %d hugepages\n", i); break; }
        ((volatile char *)v)[0] = 1; /* fault it in */
        uint64_t pa = phys_of(v);
        if (pa < boundary) lo[nlo++] = v; else hi[nhi++] = v;
    }
    printf("1 GiB pages: %d below 0x%llx (8-way region), %d at/above (4-way region)\n", nlo, (unsigned long long)boundary, nhi);
    int need = 3 * PER_ARRAY;
    if (nlo >= need) run("8-way interleaved region (low)", lo); else printf("not enough low pages\n");
    if (nhi >= need) run("4-way interleaved region (high)", hi); else printf("not enough high pages\n");
    return 0;
}
