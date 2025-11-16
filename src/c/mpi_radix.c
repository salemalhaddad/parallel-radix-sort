#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int cmp_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

/* Counting sort by the digit represented by exp (1, 10, 100, ...). */
static void count_sort(int *a, int n, int exp) {
    int count[10] = {0};
    int *output = (int *)malloc(sizeof(int) * (n > 0 ? n : 1));
    if (!output) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < n; ++i) {
        int digit = (a[i] / exp) % 10;
        count[digit]++;
    }

    for (int i = 1; i < 10; ++i) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; --i) { /* stable */
        int digit = (a[i] / exp) % 10;
        output[count[digit] - 1] = a[i];
        count[digit]--;
    }

    memcpy(a, output, sizeof(int) * n);
    free(output);
}

static void radix_sort(int *a, int n, int global_max) {
    for (int exp = 1; global_max / exp > 0; exp *= 10) {
        count_sort(a, n, exp);
    }
}

/* mpi_radix_sort_buffer: scatter -> local radix -> gather+merge (root).
   root_data is only valid on rank 0; if NULL, root generates randoms.
   root_output (rank 0 only) copies the sorted array if non-NULL. */
static int mpi_radix_sort_buffer(const int *root_data,
                                 long n,
                                 int verify,
                                 unsigned int seed,
                                 double *elapsed,
                                 int *root_output,
                                 MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* counts/displs for Scatterv/Gatherv */
    int *counts = (int *)malloc(sizeof(int) * size);
    int *displs = (int *)malloc(sizeof(int) * size);
    if (!counts || !displs) {
        fprintf(stderr, "Allocation failed for counts/displs\n");
        MPI_Abort(comm, 1);
    }

    long base = n / size;
    long rem = n % size;
    for (int r = 0; r < size; ++r) {
        counts[r] = (int)(base + (r < rem ? 1 : 0));
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }

    int local_n = counts[rank];
    int *local = (int *)malloc(sizeof(int) * (local_n > 0 ? local_n : 1));
    if (!local) {
        fprintf(stderr, "Allocation failed for local buffer\n");
        MPI_Abort(comm, 1);
    }

    int *input = NULL;
    int *gathered = NULL;
    if (rank == 0) {
        input = (int *)malloc(sizeof(int) * (n > 0 ? n : 1));
        gathered = (int *)malloc(sizeof(int) * (n > 0 ? n : 1));
        if (!input || !gathered) {
            fprintf(stderr, "Allocation failed for input/gathered buffers\n");
            MPI_Abort(comm, 1);
        }

        if (root_data) {
            memcpy(input, root_data, sizeof(int) * n);
        } else {
            srand(seed);
            for (long i = 0; i < n; ++i) {
                input[i] = rand() % 1000000000; /* 0..1e9-1 */
            }
        }
    }

    /* Broadcast seed adjustment for deterministic per-rank RNG. */
    unsigned int seed_offset = seed;
    MPI_Bcast(&seed_offset, 1, MPI_UNSIGNED, 0, comm);
    srand(seed_offset + (unsigned int)rank);

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    MPI_Scatterv(input, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, comm);

    int local_max = 0;
    for (int i = 0; i < local_n; ++i) {
        if (local[i] > local_max) local_max = local[i];
    }

    int global_max = 0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, comm);

    if (global_max > 0 && local_n > 0) {
        radix_sort(local, local_n, global_max);
    }

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, comm);

    double t1 = MPI_Wtime();

    int ok = 1;
    if (rank == 0) {
        /* k-way merge of sorted chunks into input buffer. */
        int *idx = (int *)calloc(size, sizeof(int));
        if (!idx) {
            fprintf(stderr, "Allocation failed for idx\n");
            MPI_Abort(comm, 1);
        }

        for (long out = 0; out < n; ++out) {
            int min_rank = -1;
            int min_val = 0;
            for (int r = 0; r < size; ++r) {
                if (idx[r] < counts[r]) {
                    int val = gathered[displs[r] + idx[r]];
                    if (min_rank == -1 || val < min_val) {
                        min_rank = r;
                        min_val = val;
                    }
                }
            }
            input[out] = min_val;
            idx[min_rank]++;
        }

        if (verify) {
            for (long i = 1; i < n; ++i) {
                if (input[i - 1] > input[i]) {
                    ok = 0;
                    fprintf(stderr, "Verification failed at index %ld (%d > %d)\n",
                            i, input[i - 1], input[i]);
                    break;
                }
            }
        }

        if (root_output && n > 0) {
            memcpy(root_output, input, sizeof(int) * n);
        }

        free(idx);
    }

    if (elapsed && rank == 0) {
        *elapsed = t1 - t0;
    }

    free(counts);
    free(displs);
    free(local);
    if (rank == 0) {
        free(input);
        free(gathered);
    }

    return ok;
}

static void usage(int rank) {
    if (rank == 0) {
        fprintf(stderr,
                "Usage: mpiexec -n <p> ./mpi_radix [--n <count>] [--verify] [--seed <s>] "
                "[--bench] [--correctness]\n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = 100000;   /* default input size */
    int verify = 0;
    unsigned int seed = (unsigned int)time(NULL);
    int bench = 0;
    int correctness = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench = 1;
        } else if (strcmp(argv[i], "--correctness") == 0) {
            correctness = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(rank);
            MPI_Finalize();
            return 0;
        } else {
            usage(rank);
            MPI_Finalize();
            return 1;
        }
    }

    if (n < 0) {
        if (rank == 0) {
            fprintf(stderr, "n must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Correctness tests mimic sequential.py small cases + sample of 20. */
    if (correctness) {
        int tests[][10] = {
            {0}, /* placeholder; size handled separately */
            {5},
            {3,1,2},
            {1,2,3,4},
            {4,3,2,1},
            {5,5,5,5},
            {10,0,100,7,7,3,999},
            {170,45,75,90,802,24,2,66}
        };
        int lens[] = {0,1,3,4,4,4,7,8};
        int num_tests = sizeof(lens) / sizeof(lens[0]);

        for (int t = 0; t < num_tests; ++t) {
            double elapsed = 0.0;
            int sorted_buf[10];
            int *sorted_ptr = lens[t] > 0 ? sorted_buf : NULL;
            int ok = mpi_radix_sort_buffer(t == 0 ? NULL : tests[t],
                                           lens[t],
                                           1,
                                           seed + (unsigned int)t,
                                           &elapsed,
                                           sorted_ptr,
                                           MPI_COMM_WORLD);
            if (rank == 0 && lens[t] > 0) {
                int expected[10];
                memcpy(expected, tests[t], sizeof(int) * lens[t]);
                qsort(expected, lens[t], sizeof(int), cmp_int);
                if (memcmp(expected, sorted_ptr, sizeof(int) * lens[t]) != 0) {
                    ok = 0;
                }
            }
            if (rank == 0) {
                printf("[correctness] test %d (n=%d): %s (%.6f s)\n",
                       t, lens[t], ok ? "PASS" : "FAIL", elapsed);
            }
        }

        /* Sample of 20 integers for report screenshot. */
        int sample_n = 20;
        int sample[20];
        int sorted_sample[20];
        if (rank == 0) {
            srand(seed + 12345);
            for (int i = 0; i < sample_n; ++i) {
                sample[i] = rand() % 1000;
            }
            printf("\n=== Sample of 20 integers ===\nUnsorted: ");
            for (int i = 0; i < sample_n; ++i) {
                printf("%d%s", sample[i], (i + 1 == sample_n) ? "\n" : " ");
            }
        }
        mpi_radix_sort_buffer(sample, sample_n, 1, seed + 12345, NULL, sorted_sample, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Sorted:   ");
            for (int i = 0; i < sample_n; ++i) {
                printf("%d%s", sorted_sample[i], (i + 1 == sample_n) ? "\n" : " ");
            }
        }

        MPI_Finalize();
        return 0;
    }

    /* Benchmark mode mirrors sequential/multiprocessing sizes. */
    if (bench) {
        long sizes[] = {10000, 100000, 1000000, 10000000};
        int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
        for (int i = 0; i < num_sizes; ++i) {
            double elapsed = 0.0;
            int ok = mpi_radix_sort_buffer(NULL,
                                           sizes[i],
                                           verify,
                                           seed + (unsigned int)i,
                                           &elapsed,
                                           NULL,
                                           MPI_COMM_WORLD);
            if (rank == 0) {
                printf("n = %10ld across %d ranks -> time = %.3f s%s\n",
                       sizes[i], size, elapsed, verify && !ok ? " (verify FAILED)" : "");
            }
        }
        MPI_Finalize();
        return 0;
    }

    /* Single run (default). */
    double elapsed = 0.0;
    int ok = mpi_radix_sort_buffer(NULL, n, verify, seed, &elapsed, NULL, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Sorted %ld integers across %d ranks in %.3f s.\n", n, size, elapsed);
        if (verify && !ok) {
            fprintf(stderr, "Verification failed.\n");
        }
    }

    MPI_Finalize();
    return 0;
}
