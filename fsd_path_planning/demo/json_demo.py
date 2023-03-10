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


def main(data_path: Optional[Path] = None, data_rate: float = 10) -> None:
    planner = PathPlanner(MissionTypes.trackdrive)

    positions, directions, cone_observations = load_data_json(data_path)

    paths = np.array(
        [
            planner.calculate_path_in_global_frame(
                cones,
                position,
                direction,
            )
            for (position, direction, cones) in tqdm(
                zip(positions, directions, cone_observations), total=len(positions)
            )
        ]
    )

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
    (path,) = ax.plot([], [], "r-", label="Path")
    (position,) = ax.plot([], [], "go", label="Position")

    def init():
        yellow_cones.set_data([], [])
        blue_cones.set_data([], [])
        path.set_data([], [])
        position.set_data([], [])
        return yellow_cones, blue_cones, path, position

    def animate(i):
        yellow_cones.set_data(
            cone_observations[i][ConeTypes.YELLOW][:, 0],
            cone_observations[i][ConeTypes.YELLOW][:, 1],
        )
        blue_cones.set_data(
            cone_observations[i][ConeTypes.BLUE][:, 0],
            cone_observations[i][ConeTypes.BLUE][:, 1],
        )
        path.set_data(paths[i][:, 1], paths[i][:, 2])
        position.set_data(positions[i][0], positions[i][1])
        return yellow_cones, blue_cones, path, position

    anim = matplotlib.animation.FuncAnimation(
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
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    # extract data
    data = json.loads(data_path.read_text())

    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cone_observations = [[np.array(c) for c in d["slam_cones"]] for d in data]
    return positions, directions, cone_observations


if __name__ == "__main__":
    typer.run(main)
