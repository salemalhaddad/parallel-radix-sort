"""
MPI-based radix sort using mpi4py.

Run with something like:
    mpiexec -n 4 python mpi_radix.py --n 100000 --verify
"""

from __future__ import annotations

import argparse
import heapq
import random
import time
from typing import List, Optional

from mpi4py import MPI


def count_sort(A: List[int], exp: int) -> None:
    """Counting sort by digit (exp)."""
    n = len(A)
    C = [0] * 10
    output = [0] * n

    for v in A:
        C[(v // exp) % 10] += 1

    for i in range(1, 10):
        C[i] += C[i - 1]

    # Stable traversal from the right
    for v in reversed(A):
        digit = (v // exp) % 10
        output[C[digit] - 1] = v
        C[digit] -= 1

    A[:] = output


def radix_sort(A: List[int]) -> List[int]:
    if not A or len(A) == 1:
        return A

    max_value = max(A)
    exp = 1
    while max_value // exp > 0:
        count_sort(A, exp)
        exp *= 10
    return A


def chunkify(data: List[int], chunks: int) -> List[List[int]]:
    """Split data into nearly equal chunks."""
    n = len(data)
    size = (n + chunks - 1) // chunks
    return [data[i : i + size] for i in range(0, n, size)]


def mpi_radix_sort(data: Optional[List[int]], comm: MPI.Comm = MPI.COMM_WORLD) -> Optional[List[int]]:
    """Distributed radix sort: scatter → local sort → gather+merge on rank 0."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunks = chunkify(data, size) if rank == 0 else None
    local_chunk: List[int] = comm.scatter(chunks, root=0)

    radix_sort(local_chunk)

    gathered = comm.gather(local_chunk, root=0)
    if rank != 0:
        return None

    merged = list(heapq.merge(*gathered))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI radix sort demo")
    parser.add_argument("--n", type=int, default=100_000, help="Number of random integers to sort (root generates).")
    parser.add_argument("--verify", action="store_true", help="Check the final array is sorted on rank 0.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed + rank)  # make per-rank seed deterministic but distinct

    data: Optional[List[int]]
    if rank == 0:
        data = [random.randint(0, 10**9) for _ in range(args.n)]
    else:
        data = None

    comm.barrier()
    t0 = time.time()
    sorted_data = mpi_radix_sort(data, comm=comm)
    comm.barrier()
    t1 = time.time()

    if rank == 0:
        print(f"Sorted {args.n:,} integers across {comm.Get_size()} ranks in {t1 - t0:.3f} s.")
        if args.verify and sorted_data != sorted(data or []):
            raise AssertionError("Result is not sorted correctly")


if __name__ == "__main__":
    main()
