"""Tests for test distance calculator."""

from pytest import approx as pytest_approx, raises as pytest_raises

from mage.geography import (
    InvalidCoordinatesException,
    InvalidMetricException,
    calculate_distance_between_points,
)

# Zadar and Zagreb; the calculator reads these coordinates without mutating them.
POINT_A = {"lat": 44.1194, "lng": 15.2314}
POINT_B = {"lat": 45.8150, "lng": 15.9819}


def test_distance_between_points_km():
    result = calculate_distance_between_points(POINT_A, POINT_B, metrics="km")

    assert result == pytest_approx(197.56, 0.1)


def test_distance_between_points_m():
    result = calculate_distance_between_points(POINT_A, POINT_B, metrics="m")

    assert result == pytest_approx(197568.2, 0.1)


def test_wrong_metrics():
    with pytest_raises(InvalidMetricException):
        calculate_distance_between_points(POINT_A, POINT_B, metrics="r")


def test_wrong_keys():
    point_a = {"lat": 1.0}

    with pytest_raises(InvalidCoordinatesException):
        calculate_distance_between_points(point_a, POINT_B)
