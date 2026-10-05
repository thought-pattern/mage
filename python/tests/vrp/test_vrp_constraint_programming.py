"""Tests for test vrp constraint programming."""

from numpy import array as np_array
from pytest import raises as pytest_raises

from mage.constraint_programming.vrp_cp_solver import VRPConstraintProgrammingSolver
from mage.geography import InvalidDepotException

# Symmetric route costs between three locations.
DEFAULT_DISTANCES = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]


def test_negative_depot_index_raise_exception():
    distance_matrix = np_array(DEFAULT_DISTANCES)
    with pytest_raises(InvalidDepotException):
        VRPConstraintProgrammingSolver(no_vehicles=2, distance_matrix=distance_matrix, depot_index=-1)


def test_depot_index_to_big_raise_exception():
    distance_matrix = np_array(DEFAULT_DISTANCES)
    with pytest_raises(InvalidDepotException):
        VRPConstraintProgrammingSolver(
            no_vehicles=2,
            distance_matrix=distance_matrix,
            depot_index=len(distance_matrix),
        )
