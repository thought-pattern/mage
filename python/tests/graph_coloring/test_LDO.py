"""Tests for test LDO."""

from mage.graph_coloring_module.algorithms.greedy.LDO import LDO
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter

# Five test nodes; each adjacency maps a node to its (neighbor, weight) pairs. Graph copies what it reads.
GRAPH_NODES = [0, 1, 2, 3, 4]
GRAPH_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2), (4, 5)],
    2: [(0, 3), (1, 2), (3, 3)],
    3: [(2, 3)],
    4: [(1, 5)],
}
NOT_CONNECTED_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2)],
    2: [(0, 3), (1, 2)],
    3: [(4, 3)],
    4: [(3, 3)],
}


def test_LDO():
    graph_1 = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    assert sorted(list(graph_1.nodes), key=graph_1.degree, reverse=True) == [1, 2, 0, 3, 4]
    algorithm = LDO()
    individual = algorithm.run(graph_1, {Parameter.NO_OF_COLORS: 3})

    assert individual.check_coloring()
    assert len(individual.chromosome) == len(graph_1)
    assert all(0 <= color < 3 for color in individual.chromosome)

    star = Graph([0, 1, 2, 3, 4], {0: [(1, 1), (2, 1), (3, 1), (4, 1)], 1: [(0, 1)], 2: [(0, 1)], 3: [(0, 1)], 4: [(0, 1)]})
    star_coloring = algorithm.run(star, {Parameter.NO_OF_COLORS: 2})
    assert star_coloring.check_coloring()
    assert len(set(star_coloring.chromosome[1:])) == 1
    assert star_coloring.chromosome[0] != star_coloring.chromosome[1]


def test_not_connected_graph():
    graph_not_connected = Graph(GRAPH_NODES, NOT_CONNECTED_ADJACENCY)
    algorithm = LDO()
    individual = algorithm.run(graph_not_connected, {Parameter.NO_OF_COLORS: 3})

    assert individual.check_coloring()
    assert len(individual.chromosome) == len(graph_not_connected)
    assert all(0 <= color < 3 for color in individual.chromosome)


def test_empty_graph():
    graph = Graph([], {})
    algorithm = LDO()
    individual = algorithm.run(graph, {Parameter.NO_OF_COLORS: 3})

    assert len(individual.chromosome) == 0
