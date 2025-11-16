#ifndef _WIN32
#define _XOPEN_SOURCE 700
#endif

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS)

typedef struct {
    int tid;
    int threads;
    long n;
    int *arr;
    int *tmp;
    int *counts;
    pthread_barrier_t *barrier;
} worker_ctx;

static unsigned int lcg_next(unsigned int *state) {
    *state = 1664525u * (*state) + 1013904223u;
    return *state;
}

static void fill_random(int *dst, long n, unsigned int seed) {
    unsigned int state = seed ? seed : 1u;
    for (long i = 0; i < n; ++i) {
        dst[i] = (int)(lcg_next(&state) % 1000000000u);
    }
}

static int verify_sorted(const int *arr, long n) {
    for (long i = 1; i < n; ++i) {
        if (arr[i - 1] > arr[i]) {
            return 0;
        }
    }
    return 1;
}

static int cmp_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

#ifdef _WIN32
static double wall_time(void) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}

static int default_thread_count(void) {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (int)(info.dwNumberOfProcessors ? info.dwNumberOfProcessors : 4);
}
#else
static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static int default_thread_count(void) {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs < 1) {
        nprocs = 4;
    }
    return (int)nprocs;
}
#endif

static void *radix_worker(void *arg) {
    worker_ctx *ctx = (worker_ctx *)arg;
    long chunk = (ctx->n + ctx->threads - 1) / ctx->threads;

    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        long start = ctx->tid * chunk;
        long end = start + chunk;
        if (start > ctx->n) {
            start = ctx->n;
        }
        if (end > ctx->n) {
            end = ctx->n;
        }

        int *local_counts = ctx->counts + ctx->tid * RADIX;
        memset(local_counts, 0, sizeof(int) * RADIX);
        for (long i = start; i < end; ++i) {
            unsigned int digit = ((unsigned int)ctx->arr[i] >> shift) & (RADIX - 1);
            local_counts[digit]++;
        }

        pthread_barrier_wait(ctx->barrier);

        if (ctx->tid == 0) {
            int total = 0;
            for (int digit = 0; digit < RADIX; ++digit) {
                for (int t = 0; t < ctx->threads; ++t) {
                    int idx = t * RADIX + digit;
                    int c = ctx->counts[idx];
                    ctx->counts[idx] = total;
                    total += c;
                }
            }
        }

        pthread_barrier_wait(ctx->barrier);

        for (long i = end; i-- > start;) {
            unsigned int digit = ((unsigned int)ctx->arr[i] >> shift) & (RADIX - 1);
            int pos = ctx->counts[ctx->tid * RADIX + digit]++;
            ctx->tmp[pos] = ctx->arr[i];
        }

        pthread_barrier_wait(ctx->barrier);

        for (long i = start; i < end; ++i) {
            ctx->arr[i] = ctx->tmp[i];
        }

        pthread_barrier_wait(ctx->barrier);
    }

    return NULL;
}

static double radix_sort_pthreads(int *arr, long n, int threads) {
    if (n <= 1) {
        return 0.0;
    }
    if (threads < 1) {
        threads = 1;
    }
    if (threads > n && n > 0) {
        threads = (int)n;
    }

    int *tmp = (int *)malloc(sizeof(int) * n);
    int *counts = (int *)malloc(sizeof(int) * RADIX * threads);
    pthread_t *tids = (pthread_t *)malloc(sizeof(pthread_t) * threads);
    worker_ctx *ctx = (worker_ctx *)malloc(sizeof(worker_ctx) * threads);
    if (!tmp || !counts || !tids || !ctx) {
        fprintf(stderr, "[pthread] Allocation failed\n");
        exit(1);
    }

    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, threads) != 0) {
        fprintf(stderr, "[pthread] Failed to init barrier\n");
        exit(1);
    }

    double t0 = wall_time();
    for (int t = 0; t < threads; ++t) {
        ctx[t].tid = t;
        ctx[t].threads = threads;
        ctx[t].n = n;
        ctx[t].arr = arr;
        ctx[t].tmp = tmp;
        ctx[t].counts = counts;
        ctx[t].barrier = &barrier;
        if (pthread_create(&tids[t], NULL, radix_worker, &ctx[t]) != 0) {
            fprintf(stderr, "[pthread] Failed to create thread %d\n", t);
            exit(1);
        }
    }

    for (int t = 0; t < threads; ++t) {
        pthread_join(tids[t], NULL);
    }
    double t1 = wall_time();

    pthread_barrier_destroy(&barrier);
    free(tmp);
    free(counts);
    free(tids);
    free(ctx);
    return t1 - t0;
}

static int run_random_case(long n,
                           int threads,
                           int verify,
                           unsigned int seed,
                           double *elapsed) {
    int *data = (int *)malloc(sizeof(int) * (n > 0 ? n : 1));
    if (!data) {
        fprintf(stderr, "[pthread] Allocation failed for input buffer\n");
        exit(1);
    }
    fill_random(data, n, seed);
    double t = radix_sort_pthreads(data, n, threads);
    if (elapsed) {
        *elapsed = t;
    }
    int ok = (!verify) || verify_sorted(data, n);
    free(data);
    return ok;
}

static void print_array(const int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d%s", arr[i], (i + 1 == n) ? "\n" : " ");
    }
}

static void run_correctness_suite(int threads, unsigned int seed) {
    int tests[][10] = {
        {0},
        {5},
        {3, 1, 2},
        {1, 2, 3, 4},
        {4, 3, 2, 1},
        {5, 5, 5, 5},
        {10, 0, 100, 7, 7, 3, 999},
        {170, 45, 75, 90, 802, 24, 2, 66}
    };
    int lens[] = {0, 1, 3, 4, 4, 4, 7, 8};
    int num_tests = (int)(sizeof(lens) / sizeof(lens[0]));

    for (int t = 0; t < num_tests; ++t) {
        int len = lens[t];
        int buf[10];
        memcpy(buf, tests[t], sizeof(int) * len);
        double elapsed = radix_sort_pthreads(buf, len, threads);

        int expected[10];
        memcpy(expected, tests[t], sizeof(int) * len);
        qsort(expected, len, sizeof(int), cmp_int);

        int ok = memcmp(expected, buf, sizeof(int) * len) == 0;
        printf("[correctness] test %d (n=%d): %s (%.6f s)\n",
               t, len, ok ? "PASS" : "FAIL", elapsed);
    }

    int sample_n = 20;
    int sample[20];
    int sorted_sample[20];
    fill_random(sample, sample_n, seed + 54321u);
    memcpy(sorted_sample, sample, sizeof(int) * sample_n);
    (void)radix_sort_pthreads(sorted_sample, sample_n, threads);

    printf("\n=== Sample of 20 integers ===\nUnsorted: ");
    print_array(sample, sample_n);
    printf("Sorted:   ");
    print_array(sorted_sample, sample_n);
    puts("");
}

static void run_benchmarks(int threads, int verify, unsigned int seed) {
    long sizes[] = {10000, 100000, 1000000, 10000000};
    int num_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    for (int i = 0; i < num_sizes; ++i) {
        double elapsed = 0.0;
        int ok = run_random_case(sizes[i], threads, verify, seed + (unsigned int)i, &elapsed);
        printf("n = %10ld | threads = %2d | time = %.3f s%s\n",
               sizes[i],
               threads,
               elapsed,
               verify && !ok ? " (verify FAILED)" : "");
    }
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [--n <count>] [--threads <t>] [--verify] "
            "[--seed <s>] [--bench] [--correctness]\n",
            prog);
}

int main(int argc, char **argv) {
    long n = 100000;
    unsigned int seed = (unsigned int)time(NULL);
    int verify = 0;
    int bench = 0;
    int correctness = 0;
    int threads = default_thread_count();

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench = 1;
        } else if (strcmp(argv[i], "--correctness") == 0) {
            correctness = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (n < 0) {
        fprintf(stderr, "n must be non-negative\n");
        return 1;
    }
    if (threads < 1) {
        fprintf(stderr, "threads must be >= 1\n");
        return 1;
    }

    if (correctness) {
        run_correctness_suite(threads, seed);
        return 0;
    }

    if (bench) {
        run_benchmarks(threads, verify, seed);
        return 0;
    }

    double elapsed = 0.0;
    int ok = run_random_case(n, threads, verify, seed, &elapsed);
    printf("[pthread] Sorted %ld integers with %d threads in %.3f s.\n",
           n, threads, elapsed);
    if (verify && !ok) {
        fprintf(stderr, "Verification failed.\n");
        return 1;
    }
    return 0;
}

