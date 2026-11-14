import math

import pytest

from sempalab.delay import matching_depth, span_packets


def test_span_packets():
    assert span_packets(6, 1) == 5
    assert span_packets(4, 8) == 24
    with pytest.raises(ValueError):
        span_packets(0, 1)
    with pytest.raises(ValueError):
        span_packets(3, 0)


def test_matching_depth_on_grid_points():
    depths = [1, 2, 4, 8]
    ats = [0.60, 0.65, 0.70, 0.72]
    assert matching_depth(depths, ats, 0.55) == 1
    assert matching_depth(depths, ats, 0.65) == pytest.approx(2)
    assert matching_depth(depths, ats, 0.72) == pytest.approx(8)


def test_matching_depth_interpolates_in_log_depth():
    # Halfway between 0.65 (D=2) and 0.70 (D=4) is log2 D = 1.5.
    assert matching_depth([1, 2, 4], [0.6, 0.65, 0.70], 0.675) == pytest.approx(2 ** 1.5)


def test_matching_depth_never_reached_and_unsorted_input():
    assert math.isinf(matching_depth([1, 2, 4], [0.6, 0.62, 0.63], 0.7))
    assert matching_depth([4, 1, 2], [0.70, 0.60, 0.65], 0.65) == pytest.approx(2)


def test_matching_depth_uses_first_crossing():
    # Non-monotone curve: the first upward crossing wins.
    assert matching_depth([1, 2, 4, 8], [0.6, 0.7, 0.65, 0.8], 0.68) == pytest.approx(2 ** 0.8)


def test_matching_depth_rejects_bad_input():
    with pytest.raises(ValueError):
        matching_depth([1, 2], [0.5], 0.5)
    with pytest.raises(ValueError):
        matching_depth([0, 2], [0.5, 0.6], 0.5)
