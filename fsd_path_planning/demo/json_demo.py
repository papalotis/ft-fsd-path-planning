import json
from pathlib import Path
from typing import Optional

import numpy as np

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner

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
    data_path: Optional[Path] = None,
    data_rate: float = 10,
    remove_color_info: bool = False,
) -> None:
    planner = PathPlanner(MissionTypes.trackdrive)

    positions, directions, cone_observations = load_data_json(
        data_path, remove_color_info=remove_color_info
    )

    results = []

    for i, (position, direction, cones) in tqdm(
        enumerate(zip(positions, directions, cone_observations)),
        total=len(positions),
    ):
        try:
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

    # plot animation
    fig = plt.figure()

    xmin = np.min(positions[:, 0])
    xmax = np.max(positions[:, 0])
    ymin = np.min(positions[:, 1])
    ymax = np.max(positions[:, 1])

    x_range = xmax - xmin
    y_range = ymax - ymin
    if x_range > y_range:
        ymax = ymin + x_range
    else:
        xmax = xmin + y_range

    ax = plt.axes(
        xlim=(xmin - 5, xmax + 5),
        ylim=(ymin - 5, ymax + 5),
    )
    (yellow_cones,) = ax.plot([], [], "yo", label="Yellow cones")
    (blue_cones,) = ax.plot([], [], "bo", label="Blue cones")
    (unknown_cones,) = ax.plot([], [], "ko", label="Unknown cones")
    (yellow_cones_sorted,) = ax.plot([], [], "y-", label="Yellow cones (sorted)")
    (blue_cones_sorted,) = ax.plot([], [], "b-", label="Blue cones (sorted)")
    (path,) = ax.plot([], [], "r-", label="Path")
    (position,) = ax.plot([], [], "go", label="Position")
    text = ax.text(xmin + 1, ymin + 1, "", fontsize=12)

    def init():
        yellow_cones.set_data([], [])
        blue_cones.set_data([], [])
        unknown_cones.set_data([], [])
        yellow_cones_sorted.set_data([], [])
        blue_cones_sorted.set_data([], [])
        path.set_data([], [])
        position.set_data([], [])
        text.set_text("")
        return (
            yellow_cones,
            blue_cones,
            unknown_cones,
            yellow_cones_sorted,
            blue_cones_sorted,
            path,
            position,
            text,
        )

    def animate(i):
        out = results[i]
        yellow_cones.set_data(
            cone_observations[i][ConeTypes.YELLOW][:, 0],
            cone_observations[i][ConeTypes.YELLOW][:, 1],
        )
        blue_cones.set_data(
            cone_observations[i][ConeTypes.BLUE][:, 0],
            cone_observations[i][ConeTypes.BLUE][:, 1],
        )
        unknown_cones.set_data(
            cone_observations[i][ConeTypes.UNKNOWN][:, 0],
            cone_observations[i][ConeTypes.UNKNOWN][:, 1],
        ),
        blue_cones_sorted.set_data(out[1][:, 0], out[1][:, 1])
        yellow_cones_sorted.set_data(out[2][:, 0], out[2][:, 1])

        path.set_data(out[0][:, 1], out[0][:, 2])
        position.set_data(positions[i][0], positions[i][1])
        text.set_text(f"Frame: {i}")
        return (
            yellow_cones,
            blue_cones,
            unknown_cones,
            yellow_cones_sorted,
            blue_cones_sorted,
            path,
            position,
            text,
        )

    _ = matplotlib.animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(positions),
        interval=1 / data_rate * 1000,
        blit=True,
    )

    # anim.save(..., fps=data_rate)

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
