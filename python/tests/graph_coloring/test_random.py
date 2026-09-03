"""Tests for test random."""

from pytest import fixture as pytest_fixture

from mage.graph_coloring_module import Graph, Parameter, Random


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


def test_Random(graph_1):
    algorithm = Random()
    individual = algorithm.run(graph_1, {Parameter.NO_OF_COLORS: 3})

    assert len(individual.chromosome) == len(graph_1)
    assert all(0 <= color < 3 for color in individual.chromosome)


def test_not_connected_graph(graph_not_connected):
    algorithm = Random()
    individual = algorithm.run(graph_not_connected, {Parameter.NO_OF_COLORS: 3})

    assert len(individual.chromosome) == len(graph_not_connected)
    assert all(0 <= color < 3 for color in individual.chromosome)


def test_empty_graph():
    graph = Graph([], {})
    algorithm = Random()
    individual = algorithm.run(graph, {Parameter.NO_OF_COLORS: 3})

    assert len(individual.chromosome) == 0
