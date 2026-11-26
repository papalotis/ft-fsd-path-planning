"""Unit tests for fsd_path_planning.utils.utils.Timer."""

from __future__ import annotations

import numpy as np

from fsd_path_planning.utils.utils import Timer


def _timer_with_intervals(intervals: list[float]) -> Timer:
    timer = Timer(noprint=True)
    timer.intervals = list(intervals)
    return timer


class TestTimerStatistics:
    def test_mean_is_the_mean(self) -> None:
        timer = _timer_with_intervals([1.0, 2.0, 6.0])
        assert timer.get_mean_time() == 3.0

    def test_cumulative_is_the_sum(self) -> None:
        timer = _timer_with_intervals([1.0, 2.0, 6.0])
        assert timer.get_cum_time() == 9.0

    def test_std_is_the_standard_deviation_not_the_mean(self) -> None:
        intervals = [1.0, 2.0, 6.0]
        timer = _timer_with_intervals(intervals)
        assert timer.get_std_time() == float(np.std(intervals))
        assert timer.get_std_time() != timer.get_mean_time()

    def test_std_of_identical_intervals_is_zero(self) -> None:
        timer = _timer_with_intervals([2.0, 2.0, 2.0])
        assert timer.get_std_time() == 0.0


class TestTimerReporting:
    def test_report_mean_time_does_not_raise(self, capsys) -> None:
        timer = _timer_with_intervals([0.001, 0.002])
        timer.report_mean_time()
        assert "Mean time" in capsys.readouterr().out

    def test_report_cum_time_does_not_raise(self, capsys) -> None:
        timer = _timer_with_intervals([0.001, 0.002])
        timer.report_cum_time()
        assert "Cumulative time" in capsys.readouterr().out


class TestTimerContextManager:
    def test_records_one_interval_per_block(self) -> None:
        timer = Timer(noprint=True)
        for _ in range(3):
            with timer:
                pass
        assert len(timer.intervals) == 3
        assert all(interval >= 0 for interval in timer.intervals)

    def test_reset_clears_intervals(self) -> None:
        timer = _timer_with_intervals([1.0])
        timer.reset()
        assert timer.intervals == []
