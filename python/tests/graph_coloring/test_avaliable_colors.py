"""Tests for test avaliable colors."""

from mage.graph_coloring_module import Graph, available_colors

# Five test nodes; the adjacency maps a node to its (neighbor, weight) pairs. Graph copies what it reads.
GRAPH_NODES = [0, 1, 2, 3, 4]
GRAPH_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2), (4, 5)],
    2: [(0, 3), (1, 2), (3, 3)],
    3: [(2, 3)],
    4: [(1, 5)],
}


def test_no_available_colors():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    colors = available_colors(graph, 3, [0, 1, 0, 2, 1], 2)
    expected_colors = []
    assert sorted(colors) == sorted(expected_colors)


def test_available_colors():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    colors = available_colors(graph, 3, [2, 1, 0, 2, 1], 3)
    expected_colors = [1, 2]
    assert sorted(colors) == sorted(expected_colors)
