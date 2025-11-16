# Performance Summary

Recorded timings from `docs/performance_log.txt` and derived speedups vs the sequential baseline (when available). Inputs are integers uniformly sampled in `[0, 10^9)`.

## Timing and speedup table

| n (integers) | Sequential (s) | Multiprocessing (s) | Speedup vs seq | MPI 4 ranks (s) | Speedup vs seq |
|--------------|----------------|----------------------|----------------|-----------------|----------------|
| 10,000       | 1.047          | 0.377                | 2.78×          | 0.001           | 1,047×         |
| 100,000      | 11.480         | 2.937                | 3.91×          | 0.004           | 2,870×         |
| 1,000,000    | 215.044        | 60.163               | 3.57×          | 0.028           | 7,680×         |
| 10,000,000   | –              | –                    | –              | 0.213           | –              |

Notes:
- MPI results use 4 ranks. No sequential/multiprocessing measurement was logged for 10,000,000; the graph still includes the MPI data point for scale.
- The large MPI speedups reflect both true parallel speedup and the dramatically lower constant factor in the C/MPI implementation compared to the Python baselines; consider fairer comparisons by re-running with identical language/runtime or larger node counts.

## Graphs
- `docs/img/performance_comparison.svg` — log-log plot of runtime vs input size for all implementations (includes the 10,000,000 point for MPI).

To regenerate the SVG (no external deps):
```bash
python3 scripts/generate_performance_svg.py
```
