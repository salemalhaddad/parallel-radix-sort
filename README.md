# Parallel Radix Sort

Parallel and distributed radix sort implementations in Python (sequential, multiprocessing, MPI with `mpi4py`) and C (MPI with `mpicc`). Includes small correctness checks, sample outputs for reporting, and simple benchmarking helpers.

## Project layout
- `src/python/sequential_radix.py` – baseline sequential radix sort with sanity tests and benchmarks.
- `src/python/multiprocessing_radix.py` – process-based parallel radix sort; splits input, sorts per worker, merges.
- `src/python/mpi_radix.py` – MPI radix sort in Python using `mpi4py`; scatter → local radix → gather/merge.
- `src/c/mpi_radix.c` – MPI radix sort in C with correctness/benchmark modes similar to the Python scripts.
- `docs/performance_log.txt`, `docs/img/*.png` – prior performance logs and figures.
- `docs/COSC410_Project1.2025 - Tagged.pdf` – project paper/report reference.
- `bin/mpi_radix` – sample compiled MPI binary (may need rebuild for your platform).
- `docs/performance_summary.md`, `docs/img/performance_comparison.svg` – derived timing table and generated performance chart.

## Python usage
From the repo root:
```bash
# Sequential
python src/python/sequential_radix.py

# Multiprocessing (4 workers by default)
python src/python/multiprocessing_radix.py

# MPI (Python) — adjust ranks as needed
mpiexec -n 4 python src/python/mpi_radix.py --n 100000 --verify --seed 42
```

## C MPI usage
```bash
mpicc -O2 -std=c11 -o bin/mpi_radix src/c/mpi_radix.c
mpiexec -n 4 ./bin/mpi_radix --bench --verify --seed 42
# or a single run
mpiexec -n 4 ./bin/mpi_radix --n 100000 --verify --seed 42
```
Flags:
- `--bench` runs n ∈ {10k, 100k, 1,000k} (similar to Python scripts).
- `--correctness` runs small canonical tests + prints a sample of 20 integers.
- `--verify` checks the gathered output is sorted.

## Notes
- Requires an MPI runtime (e.g., MPICH/OpenMPI). For Python MPI, install `mpi4py` in your environment.
- The current layout mirrors the testing methodology used by the sequential and multiprocessing versions: small correctness checks, sample output, and scaling benchmarks.

## Performance graphs
- Source timings are logged in `docs/performance_log.txt`.
- A derived log-log runtime plot is generated to `docs/img/performance_comparison.svg`:
  ```bash
  python3 scripts/generate_performance_svg.py
  ```
- A timing/speedup table lives in `docs/performance_summary.md`.

## Next steps
1) Add a POSIX Threads (`pthread`) radix sort implementation with the same test/bench harness.
2) Add an OpenMP version, keeping benchmark sizes aligned for fair comparison.
3) Add a Java Fork-Join version with equivalent correctness checks and benchmark sizes.
4) Unify benchmarking scripts to produce plots across all variants.
