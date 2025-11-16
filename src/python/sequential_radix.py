import random
import time

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

# ------------------ TESTING SECTION ------------------ #

def test_correctness():
    """Basic correctness tests using Python's sorted() as reference."""
    tests = [
        [],
        [5],
        [3, 1, 2],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [5, 5, 5, 5],
        [10, 0, 100, 7, 7, 3, 999],
        [170, 45, 75, 90, 802, 24, 2, 66],
    ]

    for idx, arr in enumerate(tests):
        original = list(arr)
        radix_sort(arr)
        if arr != sorted(original):
            print(f"❌ Test {idx} FAILED")
            print("  Original:", original)
            print("  Got:     ", arr)
            print("  Expected:", sorted(original))
            return
    print("✅ All basic correctness tests PASSED")


def test_sample_20():
    """Generate a sample of 20 integers: print unsorted, partial outputs, sorted."""
    A = [random.randint(0, 999) for _ in range(20)]
    print("\n=== Sample of 20 integers (for report screenshot) ===")
    print("Unsorted:", A)
    radix_sort(A)    # prints array after each digit pass
    print("Sorted:  ", A)


def benchmark_sequential():
    """Performance profiling for increasing input sizes."""
    print("\n=== Sequential Radix Sort Performance ===")
    sizes = [10_000, 100_000, 1_000_000]  # you can add 10_000_000 on a strong machine

    for n in sizes:
        A = [random.randint(0, 10**9) for _ in range(n)]
        start = time.time()
        radix_sort(A)
        end = time.time()
        print(f"n = {n:>10,}  →  time = {end - start:.3f} s")


if __name__ == "__main__":
    # 1) Basic small tests (correctness)
    test_correctness()

    # 2) Required: sample of 20 integers (unsorted → sorted + partial outputs)
    test_sample_20()

    # 3) Performance profiling up to large n (for graphs / Amdahl's Law later)
    benchmark_sequential()
    # For the full requirement (up to 10,000,000), you can also try:
    # A = [random.randint(0, 10**9) for _ in range(10_000_000)]
    # start = time.time()
    # radix_sort(A)
    # end = time.time()
    # print(f"n = 10,000,000 → time = {end - start:.3f} s")
