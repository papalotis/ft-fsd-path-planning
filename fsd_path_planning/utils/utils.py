#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A module with common utility python functions

Project: fsd_path_planning
"""

import time
from types import TracebackType
from typing import List, Optional, Type

import numpy as np
from scipy.stats import describe


class Timer:
    """
    Credit:
        https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
        but extended and improved
    """

    def __init__(self, name: str = "", noprint: bool = False) -> None:
        """Constructor for timer class

        Args:
            name (str, optional): The name of the timer. Defaults to "".
            noprint (bool, optional): If set to True print the time every time the context
            manager is exited. Defaults to False.
        """
        self.name = name
        self.print = not noprint
        self.intervals: List[float] = []
        self.start: float

    def reset(self) -> None:
        """
        Resets the timing instance. Removes all previous timings
        """
        self.intervals = []
        self.start = -1

    def __enter__(self) -> "Timer":
        """
        Start the clock

        Returns:
            Timer: self
        """
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Measure the exit time and print the difference in time since `__enter__` if
        needed
        """
        interval = time.perf_counter() - self.start
        self.intervals.append(interval)

        name = repr(self.name) if len(self.name) > 0 else ""

        space_if_name = " " if len(name) > 0 else ""

        if self.print:
            print(f"Block {name}{space_if_name}took {interval * 1_000:.4f} ms")

    def get_cum_time(self) -> float:
        """
        Gets the cumulative time spent inside this context manager. This is usefull when
        the contex manager is inside a loop and the whole time spent is desired

        Returns:
            float: The cumulative time spent inside the context manager in seconds
        """
        return sum(self.intervals)

    def get_mean_time(self) -> float:
        """
        Calculates the average of all timings

        Returns:
            float: The average time
        """
        return_value: float = np.mean(self.intervals)
        return return_value

    def get_std_time(self) -> float:
        """
        Calculates the standart devi of all timings

        Returns:
            float: The average time
        """
        return_value: float = np.mean(self.intervals)
        return return_value

    def report_cum_time(self) -> None:
        """
        Prints a formated version of get_cum_time. The time is reported in microseconds
        """
        print(
            f"Cumulative time for block {self.name} is"
            f" {self.get_cum_time() * 1_000_000:.3f} μs"
        )

    def report_mean_time(self) -> None:
        """
        Prints a formated version of get_mean_time. The time is reported in microseconds
        """
        print(
            f"Mean time for block {self.name} is"
            f" {self.get_mean_time() * 1_000_000:n.3f} μs"
        )

    def report(self) -> None:
        """
        Print a report of all the timings
        """
        print(f"Report for block {self.name}: {describe(self.intervals)}")
