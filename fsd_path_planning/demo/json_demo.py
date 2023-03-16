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
    print("Please install typer and matplotlib: pip install typer matplotlib")
    raise SystemExit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("You can get a progress bar by installing tqdm: pip install tqdm")
    tqdm = lambda x, total=None: x


def main(
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-i"),
    data_rate: float = 10,
    remove_color_info: bool = False,
    show_runtime_histogram: bool = False,
    output_path: Optional[Path] = typer.Option(None, "--output-path", "-o"),
) -> None:
    planner = PathPlanner(MissionTypes.trackdrive)

    positions, directions, cone_observations = load_data_json(
        data_path, remove_color_info=remove_color_info
    )

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
        except Exception:
            print(f"Error at frame {i}")
            raise
        results.append(out)

    if show_runtime_histogram:
        # skip the first few frames, because they include "warmup time"
        plt.hist(timer.intervals[10:])
        plt.show()

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # plot animation
    frames = []
    for i in tqdm(range(len(results)), desc="Generating animation"):
        co = cone_observations[i]
        (yellow_cones,) = plt.plot(*co[ConeTypes.YELLOW].T, "yo")
        (blue_cones,) = plt.plot(*co[ConeTypes.BLUE].T, "bo")
        (unknown_cones,) = plt.plot(*co[ConeTypes.UNKNOWN].T, "ko")
        (yellow_cones_sorted,) = plt.plot(*results[i][2].T, "y-")
        (blue_cones_sorted,) = plt.plot(*results[i][1].T, "b-")
        (path,) = plt.plot(*results[i][0][:, 1:3].T, "r-")
        (position,) = plt.plot(*positions[i], "go")
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
                yellow_cones_sorted,
                blue_cones_sorted,
                path,
                position,
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


def load_data_json(
    data_path: Optional[Path] = None,
    remove_color_info: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    # extract data
    data = json.loads(data_path.read_text())  # [:20]

    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cone_observations = [
        [np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data
    ]

    if remove_color_info:
        cones_observations_all_unkown = []
        for cones in cone_observations:
            new_observation = [np.zeros((0, 2)) for _ in ConeTypes]
            new_observation[ConeTypes.UNKNOWN] = np.row_stack(
                [c.reshape(-1, 2) for c in cones]
            )
            cones_observations_all_unkown.append(new_observation)

        cone_observations = cones_observations_all_unkown.copy()

    return positions, directions, cone_observations


if __name__ == "__main__":
    typer.run(main)
