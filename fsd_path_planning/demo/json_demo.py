from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.utils.utils import Timer

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
    from tqdm import tqdm
except ImportError:
    print("You can get a progress bar by installing tqdm: pip install tqdm")
    tqdm = lambda x, total=None: x


try:
    app = typer.Typer(pretty_exceptions_enable=False)
except TypeError:
    app = typer.Typer()


def select_mission_by_filename(filename: str) -> MissionTypes:
    is_skidpad = "skidpad" in filename

    if is_skidpad:
        print(
            'The filename contains "skidpad", so we assume that the mission is skidpad.'
        )
        return MissionTypes.skidpad

    return MissionTypes.trackdrive


def get_filename(data_path: Path | None) -> Path:
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    return data_path


@app.command()
def main(
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-i"),
    data_rate: float = 10,
    remove_color_info: bool = False,
    show_runtime_histogram: bool = False,
    output_path: Optional[Path] = typer.Option(None, "--output-path", "-o"),
) -> None:
    data_path = get_filename(data_path)

    mission = select_mission_by_filename(data_path.name)

    planner = PathPlanner(mission)

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
        planner.calculate_path_in_global_frame(
            cone_observations[0], positions[0], directions[0]
        )
    except Exception:
        print("Error during warmup")
        raise

    results = []
    timer = Timer(noprint=True)

    for i, (position, direction, cones) in tqdm(
        enumerate(zip(positions, directions, cone_observations)),
        total=len(positions),
        desc="Calculating paths",
    ):
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
            print(f"Error at frame {i}")
            raise
        results.append(out)

        if timer.intervals[-1] > 0.1:
            print(f"Frame {i} took {timer.intervals[-1]:.4f} seconds")

    if show_runtime_histogram:
        # skip the first few frames, because they include "warmup time"
        plt.hist(timer.intervals[10:])
        plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    # plot animation
    frames = []
    for i in tqdm(range(len(results)), desc="Generating animation"):
        co = cone_observations[i]
        (yellow_cones,) = plt.plot(*co[ConeTypes.YELLOW].T, "yo")
        (blue_cones,) = plt.plot(*co[ConeTypes.BLUE].T, "bo")
        (unknown_cones,) = plt.plot(*co[ConeTypes.UNKNOWN].T, "ko")
        (orange_small_cones,) = plt.plot(*co[ConeTypes.ORANGE_SMALL].T, "o", c="orange")
        (orange_big_cones,) = plt.plot(
            *co[ConeTypes.ORANGE_BIG].T, "o", c="darkorange", markersize=10
        )
        (yellow_cones_sorted,) = plt.plot(*results[i][2].T, "y-")
        (blue_cones_sorted,) = plt.plot(*results[i][1].T, "b-")
        (path,) = plt.plot(*results[i][0][:, 1:3].T, "r-")
        (position,) = plt.plot(*positions[i], "go")
        (direction,) = plt.plot(
            *np.array([positions[i], positions[i] + directions[i]]).T, "g-"
        )
        title = plt.text(
            0.5,
            1.01,
            f"Frame {i}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )
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
        anim.save(absolute_path_str, fps=data_rate)

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
            new_observation = [np.zeros((0, 2)) for _ in ConeTypes]
            new_observation[ConeTypes.UNKNOWN] = np.row_stack(
                [c.reshape(-1, 2) for c in cones]
            )
            cones_observations_all_unknown.append(new_observation)

        cone_observations = cones_observations_all_unknown.copy()

    return positions, directions, cone_observations


if __name__ == "__main__":
    app()
