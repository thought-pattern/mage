"""Tests for test SDO."""

from pytest import fixture as pytest_fixture

from mage.graph_coloring_module import SDO, Graph, Parameter


@pytest_fixture
def graph_1():
    computed_return_value = Graph(
        [0, 1, 2, 3, 4],
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 2), (4, 5)],
            2: [(0, 3), (1, 2), (3, 3)],
            3: [(2, 3)],
            4: [(1, 5)],
        },
    )
    return computed_return_value


@pytest_fixture
def graph_not_connected():
    computed_return_value = Graph(
        [0, 1, 2, 3, 4],
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 2)],
            2: [(0, 3), (1, 2)],
            3: [(4, 3)],
            4: [(3, 3)],
        },
    )
    return computed_return_value


def test_SDO(graph_1):
    algorithm = SDO()
    individual = algorithm.run(graph_1, {Parameter.NO_OF_COLORS: 3})

    assert individual.check_coloring()
    assert len(individual.chromosome) == len(graph_1)
    assert all(0 <= color < 3 for color in individual.chromosome)


def test_not_connected_graph(graph_not_connected):
    algorithm = SDO()
    individual = algorithm.run(graph_not_connected, {Parameter.NO_OF_COLORS: 3})

    assert individual.check_coloring()
    assert len(individual.chromosome) == len(graph_not_connected)
    assert all(0 <= color < 3 for color in individual.chromosome)


def test_empty_graph():
    graph = Graph([], {})
    algorithm = SDO()
    individual = algorithm.run(graph, {Parameter.NO_OF_COLORS: 3})

    assert len(individual.chromosome) == 0
