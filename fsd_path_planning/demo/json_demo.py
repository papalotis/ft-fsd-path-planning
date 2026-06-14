from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.utils.utils import Timer

RUNTIME_SUMMARY_WARMUP_FRAMES = 10
OUTLIER_IQR_SCALE = 1.5

try:
    import matplotlib.animation
    import matplotlib.pyplot as plt
    import typer
except ImportError:
    print(
        "\n\nThis demo requires matplotlib and typer to be installed. You can install"
        " them with by using the [demo] extra.\n\n"
    )
    raise

try:
    from tqdm import tqdm  # type: ignore[assignment]
except ImportError:
    print("You can get a progress bar by installing tqdm: pip install tqdm")

    def tqdm(x, total=None):  # type: ignore[misc]
        return x


try:
    app = typer.Typer(pretty_exceptions_enable=False)
except TypeError:
    app = typer.Typer()


def select_mission_by_filename(filename: str) -> MissionTypes:
    is_skidpad = "skidpad" in filename
    is_accel = "accel" in filename

    if is_skidpad:
        print(
            'The filename contains "skidpad", so we assume that the mission is skidpad.'
        )
        return MissionTypes.skidpad

    if is_accel:
        print(
            'The filename contains "accel", so we assume that the mission is acceleration.'  # noqa: E501
        )

        return MissionTypes.acceleration

    return MissionTypes.trackdrive


def get_filename(data_path: Path | None) -> Path:
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    return data_path


def print_runtime_summary(intervals: list[float]) -> None:
    if not intervals:
        print("No runtime data collected.")
        return

    runtime_ms = np.asarray(intervals, dtype=float) * 1_000
    frame_indices = np.arange(runtime_ms.size)

    def summarize(values: np.ndarray, indices: np.ndarray, label: str) -> None:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outlier_threshold = q3 + OUTLIER_IQR_SCALE * iqr
        outlier_mask = values > outlier_threshold
        outlier_indices = indices[outlier_mask]
        outlier_values = values[outlier_mask]

        print(f"{label} runtime summary ({values.size} frames):")
        print(
            "  avg="
            f"{values.mean():.2f} ms, median={np.median(values):.2f} ms, "
            f"std={values.std():.2f} ms"
        )
        print(
            "  min="
            f"{values.min():.2f} ms, p95={np.percentile(values, 95):.2f} ms, "
            f"max={values.max():.2f} ms, total={values.sum():.2f} ms"
        )

        if outlier_values.size == 0:
            print(f"  outliers: none above {outlier_threshold:.2f} ms")
            return

        sorted_outlier_order = np.argsort(outlier_values)[::-1]
        top_outliers = [
            f"frame {int(outlier_indices[idx])}={outlier_values[idx]:.2f} ms"
            for idx in sorted_outlier_order[:5]
        ]
        print(f"  outliers: {outlier_values.size} above {outlier_threshold:.2f} ms")
        print(f"  slowest outliers: {', '.join(top_outliers)}")

    print("\nRuntime overview")
    summarize(runtime_ms, frame_indices, "All frames")

    if runtime_ms.size > RUNTIME_SUMMARY_WARMUP_FRAMES:
        print(
            f"Warmup note: excluding the first {RUNTIME_SUMMARY_WARMUP_FRAMES} "
            "frames for steady-state stats."
        )
        summarize(
            runtime_ms[RUNTIME_SUMMARY_WARMUP_FRAMES:],
            frame_indices[RUNTIME_SUMMARY_WARMUP_FRAMES:],
            "Steady-state",
        )


@app.command()
def main(
    data_path: Path | None = typer.Option(None, "--data-path", "-i"),  # noqa: B008
    data_rate: float = 10,
    remove_color_info: bool = False,
    show_runtime_histogram: bool = False,
    output_path: Path | None = typer.Option(None, "--output-path", "-o"),  # noqa: B008
    experimental_performance_improvements: bool = False,
    dark_mode: bool = False,
    disable_visualization: bool = False,
) -> None:
    data_path = get_filename(data_path)

    mission = select_mission_by_filename(data_path.name)

    planner = PathPlanner(mission, experimental_performance_improvements)

    positions, directions, cone_observations = load_data_json(
        data_path, remove_color_info=remove_color_info
    )

    if not numba_cache_files_exist():
        print(
            """
It looks like this is the first time you are running this demo. It will take around a 
minute to compile the numba functions. If you want to estimate the runtime of the
planner, you should run the demo one more time after it is finished.
"""
        )

    # run planner once to "warm up" the JIT compiler / load all cached jit functions
    try:
        extra_planner = PathPlanner(mission)
        extra_planner.calculate_path_in_global_frame(
            cone_observations[0], positions[0], directions[0]
        )
    except Exception:
        print("Error during warmup")
        raise

    results = []
    timer = Timer(noprint=True)

    relocalization_info = None

    # tqdm = lambda x, desc=None, total=None: x

    for i, (position, direction, cones) in tqdm(
        enumerate(zip(positions, directions, cone_observations, strict=False)),
        total=len(positions),
        desc="Calculating paths",  # type: ignore[call-arg]
    ):
        prev_relocalization_info = relocalization_info
        relocalization_info = planner.relocalization_info
        if relocalization_info is not None and prev_relocalization_info is None:
            print("Relocalized at frame", i)
            print(f"Translation: {relocalization_info.translation.round(1)}")
            print(f"Rotation: {np.rad2deg(relocalization_info.rotation):.1f}")

        try:
            with timer:
                out = planner.calculate_path_in_global_frame(
                    cones,
                    position,
                    direction,
                    return_intermediate_results=True,
                )
        except KeyboardInterrupt:
            print(f"Interrupted by user on frame {i}")
            break
        except Exception:
            raise
        results.append(out)

        if timer.intervals[-1] > 0.1:
            print(f"Frame {i} took {timer.intervals[-1]:.4f} seconds")

    print_runtime_summary(timer.intervals)

    if disable_visualization:
        return

    if show_runtime_histogram:
        # skip the first few frames, because they include "warmup time"
        plt.hist(timer.intervals[10:])
        plt.show()

    # Conditionally apply dark mode settings
    if dark_mode:
        plt.style.use("dark_background")
        cone_colors = {
            "yellow": "yo",
            "yellow_sorted": "yo-",
            "blue": "bo",
            "blue_sorted": "bo-",
            "unknown": "wo",  # White for unknown cones in dark mode
            "orange_small": "orange",
            "orange_big": "darkorange",
            "path": "r-",
            "position": "go",
            "direction": "g-",
            "title_color": "white",
        }
    else:
        plt.style.use("default")
        cone_colors = {
            "yellow": "yo",
            "yellow_sorted": "yo-",
            "blue": "bo",
            "blue_sorted": "bo-",
            "unknown": "ko",  # Black for unknown cones in light mode
            "orange_small": "orange",
            "orange_big": "darkorange",
            "path": "r-",
            "position": "go",
            "direction": "g-",
            "title_color": "black",
        }

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    # plot animation
    frames = []

    for i in tqdm(range(len(results)), desc="Generating animation"):  # type: ignore[call-arg]
        co = cone_observations[i]

        # Use cone colors based on the mode
        (yellow_cones,) = plt.plot(
            *co[ConeTypes.YELLOW].T, cone_colors["yellow"]
        )  # Yellow cones
        (blue_cones,) = plt.plot(
            *co[ConeTypes.BLUE].T, cone_colors["blue"]
        )  # Blue cones
        (unknown_cones,) = plt.plot(
            *co[ConeTypes.UNKNOWN].T, cone_colors["unknown"]
        )  # Unknown cones
        (orange_small_cones,) = plt.plot(
            *co[ConeTypes.ORANGE_SMALL].T, "o", c=cone_colors["orange_small"]
        )  # Small orange cones
        (orange_big_cones,) = plt.plot(
            *co[ConeTypes.ORANGE_BIG].T,
            "o",
            c=cone_colors["orange_big"],
            markersize=10,  # Big orange cones
        )

        # Sorted cones and path
        (yellow_cones_sorted,) = plt.plot(
            *results[i][2].T, cone_colors["yellow_sorted"]
        )
        (blue_cones_sorted,) = plt.plot(*results[i][1].T, cone_colors["blue_sorted"])
        (path,) = plt.plot(*results[i][0][:, 1:3].T, cone_colors["path"])  # Path color

        # Position and direction
        (position,) = plt.plot(
            [positions[i][0]], [positions[i][1]], cone_colors["position"]
        )  # Position marker
        (direction,) = plt.plot(
            *np.array([positions[i], positions[i] + directions[i]]).T,
            cone_colors["direction"],  # Direction line
        )

        # Title with conditional color
        title = plt.text(
            0.5,
            1.01,
            f"Frame {i}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
            color=cone_colors["title_color"],  # Title color based on mode
        )

        # Collect the frame elements for animation
        frames.append(
            [
                yellow_cones,
                blue_cones,
                unknown_cones,
                orange_small_cones,
                orange_big_cones,
                yellow_cones_sorted,
                blue_cones_sorted,
                path,
                position,
                direction,
                title,
            ]
        )

        anim = matplotlib.animation.ArtistAnimation(
            fig, frames, interval=1 / data_rate * 1000, blit=True, repeat_delay=1000
        )

    if output_path is not None:
        absolute_path_str = str(output_path.absolute())
        typer.echo(f"Saving animation to {absolute_path_str}")
        anim.save(absolute_path_str, fps=int(data_rate))

    plt.show()


def numba_cache_files_exist() -> bool:
    package_file = Path(__file__).parent.parent
    try:
        next(package_file.glob("**/*.nbc"))
    except StopIteration:
        return False

    return True


def load_data_json(
    data_path: Path,
    remove_color_info: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    # extract data
    data = json.loads(data_path.read_text())[:]

    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cone_observations = [
        [np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data
    ]

    if remove_color_info:
        cones_observations_all_unknown = []
        for cones in cone_observations:
            new_observation: list[np.ndarray] = [np.zeros((0, 2)) for _ in ConeTypes]
            new_observation[ConeTypes.UNKNOWN] = np.row_stack(
                [c.reshape(-1, 2) for c in cones]
            )
            cones_observations_all_unknown.append(new_observation)

        cone_observations = cones_observations_all_unknown.copy()

    return positions, directions, cone_observations


if __name__ == "__main__":
    app()
