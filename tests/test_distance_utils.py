# tests/test_distance_utils.py
"""Tests for sims.utils.distance_calculation_utils module."""

import pytest
import math
import numpy as np

from sims.utils.distance_calculation_utils import (
    position_dist,
    min_l2_distance_and_target_point,
    sum_dist_path,
    dist,
    all_distances,
)


class TestPositionDist:
    """Test position_dist function."""

    def test_same_point_zero_distance(self):
        """Test that distance between same point is zero."""
        p = {"x": 1.0, "y": 2.0, "z": 3.0}
        assert position_dist(p, p) == 0.0

    def test_l2_distance_basic(self):
        """Test basic L2 distance calculation."""
        p0 = {"x": 0.0, "y": 0.0, "z": 0.0}
        p1 = {"x": 3.0, "y": 4.0, "z": 0.0}
        # sqrt(3^2 + 4^2) = 5
        assert position_dist(p0, p1, dist_fn="l2") == 5.0

    def test_l2_distance_3d(self):
        """Test L2 distance in 3D."""
        p0 = {"x": 0.0, "y": 0.0, "z": 0.0}
        p1 = {"x": 1.0, "y": 2.0, "z": 2.0}
        # sqrt(1 + 4 + 4) = 3
        assert position_dist(p0, p1, dist_fn="l2") == 3.0

    def test_l1_distance_basic(self):
        """Test basic L1 (Manhattan) distance calculation."""
        p0 = {"x": 0.0, "y": 0.0, "z": 0.0}
        p1 = {"x": 3.0, "y": 4.0, "z": 5.0}
        # |3| + |4| + |5| = 12
        assert position_dist(p0, p1, dist_fn="l1") == 12.0

    def test_ignore_y_l2(self):
        """Test L2 distance ignoring Y coordinate."""
        p0 = {"x": 0.0, "y": 0.0, "z": 0.0}
        p1 = {"x": 3.0, "y": 100.0, "z": 4.0}
        # sqrt(3^2 + 4^2) = 5 (ignoring y)
        assert position_dist(p0, p1, ignore_y=True, dist_fn="l2") == 5.0

    def test_ignore_y_l1(self):
        """Test L1 distance ignoring Y coordinate."""
        p0 = {"x": 0.0, "y": 0.0, "z": 0.0}
        p1 = {"x": 3.0, "y": 100.0, "z": 4.0}
        # |3| + |4| = 7 (ignoring y)
        assert position_dist(p0, p1, ignore_y=True, dist_fn="l1") == 7.0

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        p0 = {"x": -1.0, "y": -2.0, "z": -3.0}
        p1 = {"x": 1.0, "y": 2.0, "z": 3.0}
        # sqrt(4 + 16 + 36) = sqrt(56)
        expected = math.sqrt(56)
        assert abs(position_dist(p0, p1) - expected) < 1e-10

    def test_invalid_dist_fn_raises(self):
        """Test that invalid dist_fn raises NotImplementedError."""
        p0 = {"x": 0.0, "y": 0.0, "z": 0.0}
        p1 = {"x": 1.0, "y": 1.0, "z": 1.0}
        with pytest.raises(NotImplementedError):
            position_dist(p0, p1, dist_fn="invalid")

    @pytest.mark.parametrize(
        "p0,p1,expected",
        [
            ({"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}, 1.0),
            ({"x": 0, "y": 0, "z": 0}, {"x": 0, "y": 1, "z": 0}, 1.0),
            ({"x": 0, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 1}, 1.0),
            ({"x": 1, "y": 1, "z": 1}, {"x": 2, "y": 2, "z": 2}, math.sqrt(3)),
        ],
    )
    def test_parametrized_distances(self, p0, p1, expected):
        """Test various distance calculations."""
        assert abs(position_dist(p0, p1) - expected) < 1e-10


class TestDist:
    """Test dist function."""

    def test_same_point(self):
        """Test distance between same point."""
        loc = {"x": 1.0, "y": 2.0, "z": 3.0}
        assert dist(loc, loc) == 0.0

    def test_basic_distance(self):
        """Test basic distance calculation."""
        loc1 = {"x": 0.0, "y": 0.0, "z": 0.0}
        loc2 = {"x": 3.0, "y": 4.0, "z": 0.0}
        assert dist(loc1, loc2) == 5.0


class TestMinL2DistanceAndTargetPoint:
    """Test min_l2_distance_and_target_point function."""

    def test_single_target(self):
        """Test with single target point."""
        source = {"x": 0.0, "y": 0.0, "z": 0.0}
        targets = [{"x": 3.0, "y": 4.0, "z": 0.0}]

        min_dist, closest = min_l2_distance_and_target_point(source, targets)
        assert min_dist == 5.0
        assert closest == targets[0]

    def test_multiple_targets(self):
        """Test finding closest among multiple targets."""
        source = {"x": 0.0, "y": 0.0, "z": 0.0}
        targets = [
            {"x": 10.0, "y": 0.0, "z": 0.0},  # distance 10
            {"x": 3.0, "y": 4.0, "z": 0.0},  # distance 5
            {"x": 6.0, "y": 8.0, "z": 0.0},  # distance 10
        ]

        min_dist, closest = min_l2_distance_and_target_point(source, targets)
        assert min_dist == 5.0
        assert closest == targets[1]

    def test_source_at_target(self):
        """Test when source is at a target."""
        source = {"x": 1.0, "y": 2.0, "z": 3.0}
        targets = [
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 10.0, "y": 10.0, "z": 10.0},
        ]

        min_dist, closest = min_l2_distance_and_target_point(source, targets)
        assert min_dist == 0.0
        assert closest == targets[0]


class TestSumDistPath:
    """Test sum_dist_path function."""

    def test_empty_path(self):
        """Test with empty path."""
        assert sum_dist_path([]) == 0

    def test_single_point_path(self):
        """Test with single point path."""
        path = [{"x": 0.0, "y": 0.0, "z": 0.0}]
        assert sum_dist_path(path) == 0

    def test_two_point_path(self):
        """Test with two point path."""
        path = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 3.0, "y": 4.0, "z": 0.0},
        ]
        assert sum_dist_path(path) == 5.0

    def test_multi_point_path(self):
        """Test with multiple points."""
        path = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 3.0, "y": 4.0, "z": 0.0},  # +5
            {"x": 3.0, "y": 4.0, "z": 5.0},  # +5
        ]
        assert sum_dist_path(path) == 10.0

    def test_square_path(self):
        """Test walking a square path."""
        path = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 1.0},
            {"x": 0.0, "y": 0.0, "z": 1.0},
            {"x": 0.0, "y": 0.0, "z": 0.0},
        ]
        assert sum_dist_path(path) == 4.0


class TestAllDistances:
    """Test all_distances function."""

    def test_single_source_single_target(self):
        """Test with single source and target."""
        source = np.array([[0.0, 0.0, 0.0]])
        target = np.array([[3.0, 4.0, 0.0]])

        result = all_distances(source, target)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 5.0)

    def test_multiple_sources_and_targets(self):
        """Test with multiple sources and targets."""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

        result = all_distances(source, target)
        assert result.shape == (2, 2)
        # source[0] to target[0]: 5.0
        assert np.isclose(result[0, 0], 5.0)
        # source[0] to target[1]: 0.0
        assert np.isclose(result[0, 1], 0.0)
        # source[1] to target[0]: sqrt((3-1)^2 + 16) = sqrt(20)
        assert np.isclose(result[1, 0], np.sqrt(20))
        # source[1] to target[1]: 1.0
        assert np.isclose(result[1, 1], 1.0)

    def test_ignore_y(self):
        """Test ignoring Y coordinate."""
        source = np.array([[0.0, 100.0, 0.0]])
        target = np.array([[3.0, 200.0, 4.0]])

        result = all_distances(source, target, ignore_y=True)
        # Should only consider x and z: sqrt(9 + 16) = 5
        assert np.isclose(result[0, 0], 5.0)

    def test_same_points_zero_distance(self):
        """Test that same points have zero distance."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = all_distances(points, points)
        # Diagonal should be zeros
        assert np.allclose(np.diag(result), 0.0)
