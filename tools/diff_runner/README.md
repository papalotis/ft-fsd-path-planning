# Differential runner

Compares the planner output of two git refs, frame by frame, so that a
refactoring commit can be proven to change nothing and a deliberate algorithm
change can be quantified.

Both refs are checked out into throwaway worktrees and run in separate
subprocesses against the *same* dataset files, so the only variable is the
library code. For every frame it compares all seven stage outputs (the path and the six
intermediate results) and reports how far the planned trajectory actually
moved, plus a runtime summary and the relocalization frame.

Comparing the point arrays element by element answers the wrong question: the
path is a curve sampled at a fixed number of points, so a change that slides
the samples along the same curve looks as big as one that steers the car
somewhere else. The report therefore gives four numbers:

| Metric | Meaning |
| --- | --- |
| `lateral` | The cross-track distance: how far a point on one path lies from the nearest point on the other, measured on smooth curves through the samples and over the stretch both paths cover. It is blind to samples sliding along the same curve. This is the one that says whether the car would drive a different line, and the one the run is gated on. |
| `length` | How much longer or shorter the planned path became. The far end of the horizon is a sampling artefact, so it has a looser tolerance of its own. |
| `curvature` | The largest difference in curvature at matched positions. |
| `samples moved` | How far the sample points slid along the curve. Large here with a small `lateral` means the same trajectory, resampled. |

```bash
# the working tree against main: the gate for every refactoring commit
python tools/diff_runner/diff_runner.py --base origin/main

# one dataset only, for a fast inner loop
python tools/diff_runner/diff_runner.py --base origin/main --quick

# two commits, including the experimental sorting cache
python tools/diff_runner/diff_runner.py --base HEAD~1 --head HEAD --with-cache

# tighter gate, or no tolerance at all
python tools/diff_runner/diff_runner.py --base origin/main --tolerance 0.001
python tools/diff_runner/diff_runner.py --base origin/main --exact

# keep the collected arrays for a closer look
python tools/diff_runner/diff_runner.py --base origin/main --keep /tmp/diff
```

The default run covers all three demo datasets with and without colour
information (8,648 frames per side). `--with-cache` doubles that by also
running with `experimental_performance_improvements=True`.

Exit code is 0 when the trajectory stays within tolerance and 1 when it does
not. The default is 1 cm of lateral movement and 0.5 m of length; `--exact`
requires byte-for-byte equality instead.

When reading a large deviation, check whether the frame is one where the sorter
found anything: on frames where sorting fails the planner replays its previous
path, and two runs can drift apart there without either being better or worse.

Notes:

- `--head` defaults to `WORKTREE`, meaning the current working tree including
  uncommitted changes.
- Dataset files are taken from the base ref, so a change to the demo data can
  never show up as an algorithm difference. Override with `--datasets-from`.
- The JIT is warmed up on a throwaway planner before each run, but Numba still
  compiles new type specialisations lazily, so an occasional multi-second frame
  in the `max` column is compilation, not the algorithm.

Frames on which the sorter of a trackdrive or autocross run found no cones at
all are reported on a separate `not gated` line. On those frames the planner
only replays and extends the path it planned earlier, so any earlier
difference is extrapolated as the car keeps moving; they say nothing about how
well the code plans a track. In the bundled data they are the last frames of
the FSS log, where the car drives off the end of the mapped track. Skidpad and
acceleration never run the sorter, so all of their frames are gated.
