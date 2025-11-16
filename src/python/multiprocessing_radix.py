import multiprocessing as mp
import random
import time
import heapq

def count_sort(A, exp):
    n = len(A)

    # 0–9 digit frequencies → size 10
    C = [0] * 10

    # output array
    output = [0] * n

    # 1) Count frequency of each digit
    for i in range(n):
        digit = (A[i] // exp) % 10
        C[digit] += 1

    # 2) Convert count to cumulative count
    for i in range(1, 10):
        C[i] += C[i - 1]

    # 3) Build the output array (RIGHT → LEFT for stability)
    for i in range(n - 1, -1, -1):
        digit = (A[i] // exp) % 10
        output[C[digit] - 1] = A[i]
        C[digit] -= 1

    # 4) Copy back to A
    for i in range(n):
        A[i] = output[i]

def radix_sort(A):
    if not A or len(A) == 1:
        return A

    max_value = max(A)

    exp = 1
    while max_value / exp > 0:
        count_sort(A, exp)
        exp = exp * 10
    return A

# ------------------ PARALLEL VERSION (MULTIPROCESSING) ------------------ #
# Strategy: split array into chunks, sort each chunk with radix_sort in a
# separate process, then merge the sorted chunks.

def parallel_radix_sort(A, processes=4, verbose=False):
    if not A:
        return A

    n = len(A)
    chunk_size = (n + processes - 1) // processes  # ceil division

    # Split array into chunks
    chunks = [A[i:i + chunk_size] for i in range(0, n, chunk_size)] # task parallelism, D&C

    with mp.Pool(processes=processes) as pool: # manager-worker parallelism
        sorted_chunks = pool.map(radix_sort, chunks)

    # Merge sorted chunks
    merged = list(heapq.merge(*sorted_chunks)) # reduced, after mapped in pool.map()

    # Copy back into original list
    for i, v in enumerate(merged):
        A[i] = v

    if verbose:
        print(f"Parallel sorted array ({processes} processes): {A}")

    return A

def benchmark_parallel():
    """Performance profiling of parallel radix sort using multiprocessing."""
    print("\n=== Parallel Radix Sort Performance (multiprocessing) ===")
    sizes = [10_000, 100_000, 1_000_000]  # same sizes for fair comparison

    for n in sizes:
        A = [random.randint(0, 10**9) for _ in range(n)]
        start = time.time()
        parallel_radix_sort(A, processes=4)
        end = time.time()
        print(f"n = {n:>10,}  →  time = {end - start:.3f} s")


if __name__ == "__main__":
	benchmark_parallel()
