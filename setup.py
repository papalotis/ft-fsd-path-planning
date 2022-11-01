#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Install script for the package.
"""
from pathlib import Path

from setuptools import find_namespace_packages, setup

LICENSE = (Path(__file__).parent / "LICENSE").read_text()

setup(
    name="fsd_path_planning",
    version="0.1.0",
    packages=find_namespace_packages(),
    license=LICENSE,
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "icecream",
        "typing_extensions",
    ],
    extras_require={
        "dev": [
            "black",
            "mypy",
            "pylint",
        ],
        "demo": [
            "matplotlib",
            "typer",
            "tqdm",
            "notebook",
        ],
    },
)
