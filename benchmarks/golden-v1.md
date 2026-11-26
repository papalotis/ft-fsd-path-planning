# Runtime baseline at golden-v1

Per-frame planning time of the frozen baseline, for comparison with the
refactor. Measured with the differential runner, which warms the JIT up on one
throwaway frame and then times every frame of each run in order.

- Commit: the one tagged `golden-v1`
- Machine: Intel Xeon @ 2.10 GHz, 2 cores (cloud VM)
- Software: Python 3.13.18, NumPy 2.5.5, SciPy 1.18.1, Numba 0.68.1
- Reproduce: `python tools/diff_runner/diff_runner.py --base golden-v1 --head golden-v1 --with-cache --keep /tmp/bench`

Absolute numbers depend on the machine. Compare runs made on the same one.

| Run | Frames | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FSG'19, colour | 440 | 5.46 | 6.69 | 8.35 | 6509.2 |
| FSG'19, colour, sorting cache | 440 | 4.36 | 6.74 | 8.38 | 9.6 |
| FSG'19, no colour | 440 | 4.93 | 6.34 | 7.05 | 64.6 |
| FSG'19, no colour, sorting cache | 440 | 4.28 | 6.17 | 6.69 | 7.7 |
| FSS'19, colour | 1381 | 6.29 | 9.04 | 10.42 | 16.0 |
| FSS'19, colour, sorting cache | 1381 | 4.22 | 6.90 | 7.66 | 10.6 |
| FSS'19, no colour | 1381 | 5.69 | 7.89 | 9.05 | 12.5 |
| FSS'19, no colour, sorting cache | 1381 | 4.54 | 7.25 | 9.11 | 10.9 |
| Skidpad, colour | 341 | 1.34 | 1.94 | 11.53 | 1128.9 |
| Skidpad, colour, sorting cache | 341 | 1.32 | 2.19 | 8.94 | 13.9 |
| Skidpad, no colour | 341 | 1.31 | 1.62 | 8.46 | 18.3 |
| Skidpad, no colour, sorting cache | 341 | 1.47 | 3.08 | 9.81 | 14.8 |

The two maxima over a second are Numba compiling a new type specialisation in
the middle of a run. They are the first run of each mission in the process,
after the warm-up frame had already been planned. The warm-up only compiles
the paths that frame takes; an explicit warm-up of every jitted function is
scheduled for phase 3. Leave the max column out when comparing until then.
