#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Based on https://www.lfs.net/programmer/lyt
Load the cones from a lyt file.
"""

from pathlib import Path
from struct import Struct
from typing import Dict, cast

import numpy as np

from fsd_path_planning.utils.cone_types import ConeTypes

HEADER_STRUCT = Struct("6sBBhBB")
BLOCK_STRUCT = Struct("2h4B")

LytObjectIndexToConeType: Dict[int, ConeTypes] = {
    25: ConeTypes.UNKNOWN,
    29: ConeTypes.YELLOW,
    30: ConeTypes.YELLOW,
    23: ConeTypes.BLUE,
    24: ConeTypes.BLUE,
    27: ConeTypes.ORANGE_BIG,
    20: ConeTypes.ORANGE_SMALL,
}


def split_header_blocks(data: bytes) -> tuple[bytes, bytes]:
    """
    Split the content of the lyt file into header and block. This split is easy because
    the header has a fixed size

    Args:
        data (bytes): The content of the lyt file

    Returns:
        tuple[bytes, bytes]: The header and the block
    """
    return data[: HEADER_STRUCT.size], data[HEADER_STRUCT.size :]


def verify_lyt_header(header_data: bytes) -> None:
    """
    Parse the header and perform some sanity checks suggested by the LFS documentation

    Args:
        header_data (bytes): The header bytes of the `.lyt` file
    """

    header = cast(
        tuple[bytes, int, int, int, int, int], HEADER_STRUCT.unpack(header_data)
    )

    file_type, version, revision, _, _, _ = header
    assert file_type == b"LFSLYT"
    assert version <= 0, version
    # revision allowed up to 252
    # https://www.lfs.net/forum/thread/96153-LYT-revision-252-in-W-patch
    assert revision <= 252, revision


def lfs_heading_to_degrees(heading: int) -> float:
    """
    Convert the heading from LFS to degrees

    Args:
        heading (int): The heading as stored in the lyt file

    Returns:
        float: The heading in degrees
    """
    return heading * 360 / 256 - 180


def extract_cone_lists(
    blocks_data: bytes,
) -> tuple[list[list[tuple[float, float]]], dict[str, tuple[float, float, float]]]:
    """
    Extract the cone object positions from the object blocks bytes of a lyt file

    Args:
        blocks_data (bytes): The data in the lyt file that is not the header

    Returns:
        list[list[tuple[int, int]]]: The cone positions split by cone type
    """
    decoded_blocks = BLOCK_STRUCT.iter_unpack(blocks_data)
    all_cones_per_type: list[list[tuple[float, float]]] = [[] for _ in ConeTypes]

    start_info = {
        "position_x": None,
        "position_y": None,
        "heading_degrees": None,
    }

    # cone_info:
    for object_info in decoded_blocks:
        obj_x, obj_y, _, flags, lyt_obj_idx, heading = cast(
            tuple[int, int, int, int, int, int], object_info
        )

        try:
            cone_type = LytObjectIndexToConeType[lyt_obj_idx]
        except KeyError:
            # check if start position
            if lyt_obj_idx == 0 and flags == 0:
                heading_degrees = lfs_heading_to_degrees(heading) + 90
                position_x = obj_x / 16
                position_y = obj_y / 16

                start_info["position_x"] = position_x
                start_info["position_y"] = position_y
                start_info["heading_degrees"] = heading_degrees

            continue

        # the stored x,y pos is multiplied by
        # 16 in the file so we need to convert it back
        # (and cast to a float by using real div)
        obj_x_meters = obj_x / 16
        obj_y_meters = obj_y / 16
        all_cones_per_type[cone_type].append((obj_x_meters, obj_y_meters))

    return all_cones_per_type, start_info


def load_lyt_file(
    filename: Path | str,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """
    Load a `.lyt` file and return the positions of the cone objects inside it split
    according to `ConeTypes`

    Args:
        filename (Path): The path to the `.lyt` file

    Returns:
        list[np.ndarray]: A list of 2d np.ndarrays representing the cone positions of
        for all cone types
    """
    if isinstance(filename, str):
        filename = Path(filename)
    assert filename.is_file(), filename
    assert filename.suffix == ".lyt", filename
    data = filename.read_bytes()
    header_data, blocks_data = split_header_blocks(data)
    verify_lyt_header(header_data)

    all_cones_per_type, start_info = extract_cone_lists(blocks_data)

    all_cones_per_type_arrays = [
        np.array(cone_list).reshape(-1, 2) for cone_list in all_cones_per_type
    ]

    center = np.mean(all_cones_per_type_arrays[ConeTypes.ORANGE_BIG], axis=0)

    cone_positions_centered = [x - center for x in all_cones_per_type_arrays]

    start_info["position_x"] -= center[0]
    start_info["position_y"] -= center[1]

    return cone_positions_centered, start_info


def main():
    # parse arguments
    import argparse
    import json

    # one argument for the lyt file
    parser = argparse.ArgumentParser(description="Parse a lyt file")
    parser.add_argument("lyt_file", type=str, help="The lyt file to parse")

    args = parser.parse_args()

    # load the lyt file
    lyt_file = Path(args.lyt_file)

    cone_positions_centered, start_info = load_lyt_file(lyt_file)

    out = {
        "unknown": cone_positions_centered[ConeTypes.UNKNOWN].tolist(),
        "yellow": cone_positions_centered[ConeTypes.YELLOW].tolist(),
        "blue": cone_positions_centered[ConeTypes.BLUE].tolist(),
        "orange_small": cone_positions_centered[ConeTypes.ORANGE_SMALL].tolist(),
        "orange_big": cone_positions_centered[ConeTypes.ORANGE_BIG].tolist(),
        "start_info": start_info,
    }

    # print the cones as json
    print(json.dumps(out, indent=1))

    # visualize the cones
    # visualize_cones(out)


def visualize_cones(
    out: dict[str, list[tuple[float, float]] | dict[str, tuple[float, float, float]]],
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for v in out.values():
        if isinstance(v, dict):
            start_info = v

            heading_rad = np.deg2rad(start_info["heading_degrees"])

            n = 10

            ax.arrow(
                start_info["position_x"],
                start_info["position_y"],
                np.cos(heading_rad) * n,
                np.sin(heading_rad) * n,
                head_width=1.5,
                head_length=1,
                fc="k",
                ec="k",
            )

            # ax.plot(
            #     [start_info["position_x"], start_n_meter_forward[0]],
            #     [start_info["position_y"], start_n_meter_forward[1]],
            #     "k",
            # )

            continue

        cones = np.array(v).reshape(-1, 2)
        ax.plot(cones[:, 0], cones[:, 1], ".")

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
