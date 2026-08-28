"""Tests for test SDO."""

from random import seed as random_seed

from pytest import fixture as pytest_fixture

from mage.graph_coloring_module import SDO, Graph, Parameter


@pytest_fixture
def set_seed():
    random_seed(42)
    return False


@pytest_fixture
def graph_1():
    _return_value = Graph(
        [0, 1, 2, 3, 4],
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 2), (4, 5)],
            2: [(0, 3), (1, 2), (3, 3)],
            3: [(2, 3)],
            4: [(1, 5)],
        },
    )
    return _return_value


@pytest_fixture
def graph_not_connected():
    _return_value = Graph(
        [0, 1, 2, 3, 4],
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 2)],
            2: [(0, 3), (1, 2)],
            3: [(4, 3)],
            4: [(3, 3)],
        },
    )
    return _return_value


def test_SDO(set_seed, graph_1):
    algorithm = SDO()
    individual = algorithm.run(graph_1, {Parameter.NO_OF_COLORS: 3})

    expected_result = [2, 0, 1, 2, 1]
    assert individual.chromosome == expected_result
    return False


def test_not_connected_graph(set_seed, graph_not_connected):
    algorithm = SDO()
    individual = algorithm.run(graph_not_connected, {Parameter.NO_OF_COLORS: 3})

    expected_result = [2, 0, 1, 2, 1]
    assert individual.chromosome == expected_result
    return False


def test_empty_graph(set_seed):
    graph = Graph([], {})
    algorithm = SDO()
    individual = algorithm.run(graph, {Parameter.NO_OF_COLORS: 3})

    expected_result = []
    assert individual.chromosome == expected_result
    return False
