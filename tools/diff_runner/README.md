# Differential runner

Compares the planner output of two git refs, frame by frame, so that a
refactoring commit can be proven to change nothing and a deliberate algorithm
change can be quantified.

Both refs are checked out into throwaway worktrees and run in separate
subprocesses against the *same* dataset files, so the only variable is the
library code. For every frame it compares all seven stage outputs (the path and
the six intermediate results) with exact equality, and reports the number of
differing frames, the maximum absolute difference and the first frame that
differs, plus a runtime summary and the relocalization frame.

```bash
# the working tree against main: the gate for every refactoring commit
python tools/diff_runner/diff_runner.py --base origin/main

# one dataset only, for a fast inner loop
python tools/diff_runner/diff_runner.py --base origin/main --quick

# two commits, including the experimental sorting cache
python tools/diff_runner/diff_runner.py --base HEAD~1 --head HEAD --with-cache

# keep the collected arrays for a closer look
python tools/diff_runner/diff_runner.py --base origin/main --keep /tmp/diff
```

The default run covers all three demo datasets with and without colour
information (8,648 frames per side). `--with-cache` doubles that by also
running with `experimental_performance_improvements=True`.

Exit code is 0 when the refs agree exactly and 1 when they differ.

Notes:

- `--head` defaults to `WORKTREE`, meaning the current working tree including
  uncommitted changes.
- Dataset files are taken from the base ref, so a change to the demo data can
  never show up as an algorithm difference. Override with `--datasets-from`.
- The JIT is warmed up on a throwaway planner before each run, but Numba still
  compiles new type specialisations lazily, so an occasional multi-second frame
  in the `max` column is compilation, not the algorithm.
