# Stabilization Plan

This document tracks backward-compatible, high-impact hardening work for the library.
The focus is stability, clearer failure modes, and safer maintenance for downstream teams.

## Principles

- Preserve behavior for valid existing callers.
- Prefer validation, tests, CI, and documentation before algorithm changes.
- Treat public API compatibility as a release requirement.
- Keep phases small enough to land independently.

## Guardrails

- [ ] Run the existing regression suite before and after each phase.
- [ ] Avoid algorithmic behavior changes unless explicitly scoped and covered by regression tests.
- [ ] Keep deprecated APIs working until their removal is planned in a separate release note.
- [ ] Document any intentional behavior change in `README.md` and changelog/release notes.

## Phase 0: Baseline Safety Net

Goal: make compatibility regressions visible before touching runtime behavior.

- [ ] Add GitHub Actions CI for tests, lint, and type checking.
- [ ] Run `nox -s tests`, `nox -s lint`, and `nox -s typecheck` in CI.
- [ ] Ensure CI covers the supported Python versions already declared in `noxfile.py`.
- [ ] Decide whether slow regression tests run on every PR or on a separate trigger/schedule.
- [ ] Document the required local validation commands for contributors.

Exit criteria:

- [ ] Every PR gets automated validation.
- [ ] The main compatibility checks are reproducible locally and in CI.

## Phase 1: Public API Input Validation

Goal: fail early and predictably at the public entry points instead of failing deep in geometry code.

- [x] Add explicit validation in `PathPlanner.calculate_path_in_global_frame()`.
- [x] Validate `vehicle_position` shape and finiteness.
- [x] Validate `vehicle_direction` accepted forms and finiteness after normalization.
- [x] Validate `cones` container shape and per-cone-array `Nx2` shape.
- [x] Validate cone arrays are numeric and reject unsupported dtypes cleanly.
- [x] Replace implicit deep failures from internal `assert` paths with user-facing `TypeError` or `ValueError` at the boundary where feasible.
- [x] Keep accepted valid inputs fully backward compatible.

Exit criteria:

- [x] Invalid public inputs fail with clear, stable exceptions.
- [x] Valid inputs keep existing behavior and pass regression tests unchanged.

## Phase 2: Configuration Validation

Goal: catch invalid tuning values when configs are created, not after they reach NumPy, SciPy, or Numba code.

- [x] Add `__post_init__` validation to `SortingConfig`.
- [x] Add `__post_init__` validation to `MatchingConfig`.
- [x] Add `__post_init__` validation to `PathConfig`.
- [x] Add `__post_init__` validation to `SkidpadConfig`.
- [x] Define allowed ranges for distances, angles, horizon lengths, smoothing, and counts.
- [x] Ensure `PipelineConfig.from_json()` still gives clear errors for invalid config values.
- [x] Confirm defaults remain valid and unchanged.

Exit criteria:

- [x] Invalid config values fail immediately with actionable messages.
- [x] Existing defaults and documented examples still construct successfully.

## Phase 3: Compatibility and Validation Tests

Goal: lock in the old and new API contracts before more cleanup work lands.

- [x] Add tests for invalid public inputs in the full pipeline.
- [x] Cover wrong cone shapes, wrong vehicle vector shapes, NaN, and Inf inputs.
- [x] Add tests for accepted scalar-angle vehicle direction inputs.
- [x] Add tests for deprecated constructor parameters still working with warnings.
- [x] Add tests for deprecated `set_new_input()` flows still working with warnings.
- [x] Add tests that the new dataclass-based APIs do not emit deprecation warnings.
- [x] Keep regression tests for valid inputs unchanged.

Exit criteria:

- [x] Invalid-input behavior is intentionally specified by tests.
- [x] Deprecation paths are protected against accidental breakage.

## Phase 4: Targeted Low-Coverage Hardening

Goal: increase confidence where the code is most fragile, not by chasing a global coverage number.

- [x] Add focused tests for `sorting_cones/trace_sorter/line_segment_intersection.py`.
- [x] Add focused tests for `sorting_cones/trace_sorter/end_configurations.py`.
- [x] Add focused tests for `cone_matching/match_directions.py`.
- [x] Add focused tests for `relocalization/acceleration/acceleration_relocalization.py`.
- [x] Review whether each module needs boundary-condition tests, property-style tests, or regression fixtures.
- [x] Prefer surgical tests around geometry edge cases over broad snapshot expansion.
- [x] Re-check coverage after each targeted test addition and stop when confidence is sufficient.

Exit criteria:

- [x] The weakest algorithmic modules have direct tests for their failure-prone edge cases.
- [ ] Coverage increases in the identified hotspots without destabilizing the suite.

## Phase 5: Public Contract Documentation

Goal: reduce accidental misuse by documenting the API contract where users actually look.

- [ ] Expand `PathPlanner.calculate_path_in_global_frame()` docstring with exact accepted input shapes.
- [ ] Document cone ordering and expected `ConeTypes` mapping in the public API docs.
- [ ] Document the accepted vehicle direction formats: vector and scalar angle.
- [ ] Document which exceptions are raised for invalid public inputs.
- [ ] Update `README.md` usage examples to match the validated contract.
- [ ] Add a short migration note for deprecated constructor and `set_new_input()` flows.

Exit criteria:

- [ ] A downstream team can integrate the library without reading internal implementation files.
- [ ] The README and docstrings match the actual validated runtime behavior.

## Phase 6: Dependency and Logging Hygiene

Goal: reduce avoidable runtime surface area after the core stability work is finished.

- [ ] Audit runtime use of `loguru` and decide whether standard-library logging is sufficient.
- [ ] Remove `icecream` from runtime dependencies if it is no longer used by library code.
- [ ] Reconfirm that dependency removals do not affect public behavior or examples.
- [ ] Update packaging metadata and contributor docs if dependencies change.

Exit criteria:

- [ ] Runtime dependencies reflect actual library needs.
- [ ] Logging behavior is intentional and appropriate for a reusable library.

## Suggested Order of Execution

1. Phase 0
2. Phase 1
3. Phase 3
4. Phase 2
5. Phase 4
6. Phase 5
7. Phase 6

Rationale:

- CI and compatibility tests should land before deeper hardening.
- Public input validation should be specified by tests before broader cleanup.
- Config validation is safest once the compatibility safety net is already in place.
- Dependency cleanup should happen last because it is lower impact than contract hardening.

## Tracking Notes

- Use one PR per phase when possible.
- If a phase grows beyond a few focused commits, split it.
- Record any backward-compatibility concern directly in this file before implementation.