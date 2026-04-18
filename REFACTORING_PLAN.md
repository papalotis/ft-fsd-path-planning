# Refactoring Plan

This document catalogues concrete issues in the codebase and proposes fixes grouped into
three goals: **modularity**, **architectural consistency**, and **parameter tunability**.
Each item is actionable and scoped. Items are roughly ordered by impact within each section.

> **Implementation order:** Strictly phased (0→1→2→3). All regression tests must be green
> before advancing to the next phase.

> **Backward compatibility:** Keep old `PathPlanner(mission)` API working with deprecation
> warnings. New API is the recommended path.

---

## 1. Parameter Tunability

### 1.1 Centralize all parameters into a single, hierarchical configuration dataclass

**Problem:** Parameters are scattered across three layers and are hard to discover or override:

| Where | Examples |
|---|---|
| `config.py` factory functions | `max_n_neighbors=5`, `max_dist=6.5`, `smoothing=0.2`, `mpc_path_length=20` |
| Hardcoded in algorithm code | `car_size=2.1` in `find_configs_and_scores.py:108`, `radius=1000` in `path_calculator_helpers.py:64`, `threshold_distance=3` in `cost_function.py:254` |
| Hardcoded in skidpad relocalizer | `7.625` (circle radius), `18.25` (center distance), `eps=3` (DBSCAN), `2.4` (mean distance) in `skidpad_relocalizer.py:67-68` |

**Decision:** Create a `PipelineConfig` dataclass tree with nested `SortingConfig`, `MatchingConfig`, `PathConfig`. Add `from_json()`/`to_json()` methods for file-based configuration. Provide `default_config(mission: MissionTypes)` factory that returns per-mission defaults. Surface **all** magic numbers to config — not just user-facing ones.

Skidpad-specific constants (`SkidpadConfig`) live in a **separate** config object, not inside `PipelineConfig`.

### 1.2 Extract magic cost-function weights

**Problem:** In [`cost_function.py:283`](fsd_path_planning/sorting_cones/trace_sorter/cost_function.py), the cost weights are an inline array:
```python
factors = np.array([1000.0, 200.0, 5000.0, 1000.0, 0.0, 1000.0, 1000.0])
```
There is no way to tune them without editing source code, and the `0.0` effectively disables `change_of_direction_cost` silently.

**Decision:** Create a nested `CostWeights` dataclass within `SortingConfig`:
```python
@dataclass
class CostWeights:
    angle: float = 1000.0
    residual_distance: float = 200.0
    n_cones: float = 5000.0
    initial_direction: float = 1000.0
    direction_change: float = 0.0
    cones_on_side: float = 1000.0
    wrong_direction: float = 1000.0
```

### 1.3 Extract hardcoded thresholds in `end_configurations.py`

**Problem:** `neighbor_bool_mask_can_be_added_to_attempt()` uses:
- `4.0` — distance under which directional-angle constraint is relaxed
- `np.deg2rad(40)` inline threshold for angle checks (duplicates the config value)

**Decision:** Surface both to `SortingConfig` fields.

### 1.4 Extract path extension constants

**Problem:** In `core_calculate_path.py:296-300`, path extension uses:
- `radius_to_use = min(max(radius, 10), 100)` — min/max radius for circular arc
- `radius_to_use < 80` — branching threshold
- `np.arange(30)` — number of straight extension points
- `-20:` — number of tail points for circle fit

**Decision:** Surface all to `PathConfig` fields.

### 1.5 Extract cone matching geometry constants

**Problem:** In `core_cone_matching.py:104`:
```python
major_radius = self.state.max_search_range * 1.5
```
The `1.5` multiplier is undocumented and not configurable.

Similarly, in `functional_cone_matching.py`, the angle threshold `np.deg2rad(85)` for removing sharp-angle cones after virtual insertion is hardcoded.

**Decision:** Surface both to `MatchingConfig` fields.

### 1.6 Make the skidpad geometry configurable

**Problem:** `skidpad_relocalizer.py` has skidpad-specific constants hardcoded:
- Circle radius: `7.625`
- Center-to-center distance: `18.25`
- DBSCAN epsilon: `3`
- Mean cone distance: `2.4`, tolerance: `1.5`
- Residual threshold: `0.4`

**Decision:** Create a separate `SkidpadConfig` dataclass (not part of `PipelineConfig`). `SkidpadRelocalizer` accepts it in its constructor with defaults matching current FSG rules.

---

## 2. Modularity

### 2.1 ✅ Break up the `CalculatePath` god class (563 lines)

**Problem:** `CalculatePath` handles six distinct responsibilities:
1. Selecting which side to use for path basis
2. Computing centerline from matches
3. Fitting splines
4. Extending short paths with circular arcs
5. MPC parameterization (curvature, resampling)
6. Storing path history for fallback

**Decision:** Extract into focused helper classes:
- `PathBasisSelector` — side selection + centerline computation (methods `select_side_to_use`, `calculate_centerline_points_of_matches`)
- `PathExtender` — `connect_path_to_car`, `extend_path`, `remove_path_behind_car`, `remove_path_not_in_prediction_horizon`
- `PathParameterizer` — already partially exists, but `do_all_mpc_parameter_calculations` orchestration should move there
- Keep `CalculatePath` as a thin orchestrator

### 2.2 ✅ Break up `neighbor_bool_mask_can_be_added_to_attempt()` (~120 lines)

**Problem:** This single Numba-jitted function in `end_configurations.py` does:
1. Ellipse containment check
2. Vehicle side verification
3. Between-cone skip detection
4. Directional angle thresholding
5. Direction change detection

**Decision:** Extract each check into a separate `@my_njit` function and compose them in the main function.

### 2.3 ✅ Decouple `PathPlanner` from concrete module construction

**Problem:** `PathPlanner.__init__` directly calls `create_default_sorting()`, `create_default_cone_matching_with_non_monotonic_matches()`, and `create_default_pathing()`. You cannot inject a custom sorting or matching implementation.

**Decision:** Accept optional pre-built instances in the constructor:
```python
def __init__(self, mission, *, cone_sorting=None, cone_matching=None, pathing=None):
```
Fall back to defaults when `None`. This enables testing with mocks and swapping algorithms.

### 2.4 ✅ Remove module-level global caches

**Problem:** Four global mutable variables are used for caching:
- `adjacency_matrix.py`: `LAST_MATRIX_CALC_HASH`, `LAST_MATRIX_CALC_DISTANCE_MATRIX`
- `adjacency_matrix.py`: `LAST_IDXS_HASH`, `LAST_IDXS_CALCULATED`

These are not thread-safe and leak state between independent pipeline instances.

**Decision:** Move caches into `TraceSorter` as instance attributes. Pass them through function arguments.

### 2.5 ✅ Make `nearby_cone_search.py` caches instance-scoped

**Problem:** Similar to 2.4, `nearby_cone_search.py` uses Numba typed-dict caches (`create_search_directions_cache()`, `create_angle_cache()`) that are created per call but the pattern encourages global reuse.

**Decision:** Attach caches to the `TraceSorter` instance and pass them as arguments.

### 2.6 ✅ Separate relocalization from the main pipeline flow

**Problem:** `PathPlanner.calculate_path_in_global_frame()` has a large `if self.relocalizer is not None` branch that skips sorting and matching entirely. The relocalized vs. non-relocalized code paths are interleaved.

**Decision:** Subclass `PathPlanner`:
- `StandardPathPlanner` — the sorting→matching→path flow (default for trackdrive/autocross)
- `RelocalizingPathPlanner` — the relocalization flow (for skidpad/acceleration)

A factory function selects the right subclass based on mission.

### 2.7 ✅ Extract `circle_fit` protection

**Problem:** `circle_fit` divides by zero when given collinear points (`det == 0`). This crashes the pipeline on perfectly straight cone arrangements.

**Decision:** Add epsilon to the denominator in `circle_fit` to prevent division by zero:
```python
det = x * x - x * Mz + Cov_xy
det = det if abs(det) > 1e-15 else 1e-15  # prevent division by zero for collinear points
```

---

## 3. Architectural Consistency

### 3.1 ✅ Unify the input/state pattern across modules

**Problem:** Each module uses a different variation of the input→state→run pattern:

| Module | Pattern |
|---|---|
| `ConeSorting` | `set_new_input()` → `transition_input_to_state()` (called inside `run_cone_sorting()`) → `run_cone_sorting()` |
| `ConeMatching` | `set_new_input()` → `transition_input_to_state()` (called inside `run_cone_matching()`) → `run_cone_matching()` |
| `CalculatePath` | `set_new_input()` → `run_path_calculation()` (no explicit transition, reads `self.input` directly) |

**Decision:** Input as `run()` argument. No `set_new_input()`. Modules are stateless per call (only history buffers are retained):
```python
def run_cone_sorting(self, input: ConeSortingInput) -> SortingResult:
```
Keep the old `set_new_input()` + `run_*()` API as deprecated wrappers.

### 3.2 ✅ Standardize error handling

**Problem:** Three different error strategies:

| Strategy | Where |
|---|---|
| Custom exception (`NoPathError`) | `end_configurations.py` — raised then caught in `TraceSorter.sort_left_right()` |
| Bare `except Exception` + fallback | `core_calculate_path.py:231` — catches spline fitting failure, uses previous path |
| Silent fallback | `core_calculate_path.py:542-547` — too few cones, silently uses previous path |
| `print(e)` + re-raise | `core_calculate_path.py:254` — debug print then crash |

**Decision:** Use **loguru** (added as a main dependency in `pyproject.toml`). Replace all `print()`, `ic()`, and debug statements with `logger.debug()` / `logger.warning()`. Policy:
1. Algorithm failures with fallbacks → `logger.debug("...", exc_info=True)`, return fallback
2. Unrecoverable errors → raise typed exception (`PathPlanningError`)

### 3.3 ✅ Unify return types

**Problem:** Inconsistent return structures:
- `run_cone_sorting()` → `Tuple[FloatArray, FloatArray]`
- `run_cone_matching()` → `Tuple[FloatArray, FloatArray, IntArray, IntArray]`
- `run_path_calculation()` → `Tuple[FloatArray, FloatArray]` (but the second is the centerline basis, not documented)
- `calculate_path_in_global_frame()` → `FloatArray` or 7-tuple depending on flag

**Fix:** Use named result dataclasses:
```python
@dataclass
class SortingResult:
    left_cones: FloatArray
    right_cones: FloatArray

@dataclass
class MatchingResult:
    left_cones_with_virtual: FloatArray
    right_cones_with_virtual: FloatArray
    left_to_right_matches: IntArray
    right_to_left_matches: IntArray

@dataclass
class PathResult:
    final_path: FloatArray          # (N, 4)
    centerline_basis: FloatArray    # (M, 2)
```

### 3.4 ✅ Remove dead code and debug artifacts

**Problem:** Multiple files contain commented-out code, debug prints, and unused imports:

| File | Issue |
|---|---|
| `core_calculate_path.py` | `# ic(center_x, ...)`, `# print(repr(...))`, `# assert 0` |
| `full_pipeline.py` | `# print("prev", ...)`, `# print("trans", ...)`, `# assert 0` |
| `core_trace_sorter.py` | Commented-out `mask_cones_close` block, commented-out import |
| `end_configurations.py` | `# my_njit = lambda x: x  # XXX: just for debugging` |
| `functional_cone_matching.py` | `from icecream import ic  # noqa: F401` — imported but unused |
| `cost_function.py:302` | Duplicate `return` statement: `return final_costs.sum(axis=-1)` appears twice |
| `cost_function.py:282` | `not timer_no_print and print()` — side-effect expression used as statement |

**Decision:** Remove everything in one bulk pass.

### 3.5 Standardize constructor parameter passing

**Problem:** `ConeSorting` duplicates all constructor parameters — they're passed to the constructor, stored in `ConeSortingState`, *and* forwarded individually to `TraceSorter`:
```python
self.state = ConeSortingState(max_n_neighbors=max_n_neighbors, ...)
self.trace_sorter = TraceSorter(self.state.max_n_neighbors, self.state.max_dist, ...)
```

`CalculatePath` splits parameters between `PathCalculationScalarValues` and `SplineFitterFactory` with no clear grouping.

**Decision:** Each module accepts its config dataclass directly:
```python
class ConeSorting:
    def __init__(self, config: SortingConfig):
```
The config is the single source of truth. No parameter duplication.

### 3.6 ✅ Fix the `PathPlanner.relocalization_info` property bug

**Problem:** In `full_pipeline.py`, the `calculate_path_in_global_frame` method references `self.relocalization_info` but the property checks `self.relocalizer.is_relocalized`. If the relocalizer exists but is not yet relocalized, the property returns `None`, but the method still executed the relocalization code path (the `if self.relocalizer is not None` branch), creating a logical inconsistency — the path transform block checks `self.relocalization_info` not `self.relocalizer.is_relocalized`.

**Decision:** Fix — use consistent check (`self.relocalizer is not None and self.relocalizer.is_relocalized`).

### 3.7 ✅ Fix type annotations on the `Relocalizer` abstract class

**Problem:** `transform_to_known_map_frame` has an incorrect return type:
```python
def transform_to_known_map_frame(...) -> Tuple[RelocalizationCallable, RelocalizationCallable]:
```
It actually returns `Tuple[FloatArray, float]` (position + yaw).

**Decision:** Fix the annotation.

### 3.8 ✅ Unify naming conventions

**Problem:** Inconsistent naming across the codebase:
- `slam_position` vs `position_global` vs `vehicle_position` vs `car_pos` — all mean the same thing
- `slam_direction` vs `direction_global` vs `vehicle_direction` vs `car_dir`
- `slam_cones` vs `cones_by_type_array` vs `cones`
- `mpc_path_length` vs `maximal_distance_for_valid_path` — different abstraction levels of naming

**Decision:** Standardize on `vehicle_position`, `vehicle_direction`, `cones_by_type` everywhere.

---

## 4. Implementation Order

| Phase | Items | Risk | Benefit |
|---|---|---|---|
| **Phase 0: Cleanup** | 3.4, 3.6, 3.7, 2.7 | Very low | Removes bugs, dead code, noise |
| **Phase 1: Config** | 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.5 | Low | All parameters tunable without code changes |
| **Phase 2: Consistency** | 3.1, 3.2, 3.3, 3.8 | Medium | Uniform patterns, loguru logging, named results |
| **Phase 3: Modularity** | 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 | Medium-High | Testable units, swappable components, DI |

Each phase must keep all regression tests green before advancing to the next.
