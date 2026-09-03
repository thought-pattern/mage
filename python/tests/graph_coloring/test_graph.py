"""Tests for test graph."""

from pytest import fixture as pytest_fixture
from pytest import mark as pytest_mark

from mage.graph_coloring_module import Graph


@pytest_mark.parametrize(
    "node, neighs",
    [(0, [1, 2]), (1, [0, 3, 4]), (2, [0, 4]), (3, [1, 4]), (4, [1, 2, 3])],
)
def test_correct_get_neighbors(graph: Graph, node: int, neighs: list[int]):
    for n in graph.neighbors(node):
        assert n in neighs
    for n in neighs:
        assert n in graph.neighbors(node)


@pytest_mark.parametrize(
    "node, neighs",
    [(0, [1, 2]), (1, [0, 3, 4]), (2, [0, 4]), (3, [1, 4]), (4, [1, 2, 3])],
)
def test_correct_get_neighbors_with_mapping(graph_string_labels: Graph, node: int, neighs: list[int]):
    for n in graph_string_labels.neighbors(node):
        assert n in neighs
    for n in neighs:
        assert n in graph_string_labels.neighbors(node)


@pytest_mark.parametrize(
    "node, weight_nodes",
    [
        (0, [(1, 2), (2, 1)]),
        (1, [(0, 2), (3, 3), (4, 1)]),
        (2, [(0, 1), (4, 4)]),
        (3, [(1, 3), (4, 1)]),
        (4, [(1, 1), (2, 4), (3, 1)]),
    ],
)
def test_correct_get_weighted_neighbors(graph: Graph, node: int, weight_nodes: list[tuple[int, float]]):
    for n in graph.weighted_neighbors(node):
        assert n in weight_nodes
    for n in weight_nodes:
        assert n in graph.weighted_neighbors(node)


@pytest_mark.parametrize("node_1, node_2, weight", [(0, 1, 2), (1, 0, 2), (2, 4, 4), (3, 1, 3), (0, 4, 0)])
def test_correct_get_weight(graph: Graph, node_1: int, node_2: int, weight: float):
    assert graph.weight(node_1, node_2) == weight


@pytest_mark.parametrize(
    "node, label",
    [
        (0, "0"),
        (1, "1"),
        (2, "2"),
    ],
)
def test_correct_get_label(graph_string_labels: Graph, node: int, label: object):
    assert label == graph_string_labels.label(node)


def test_correct_number_of_nodes(graph: Graph):
    assert graph.number_of_nodes() == 5


def test_correct_number_of_edges(graph: Graph):
    assert graph.number_of_edges() == 6


def test_correct_length_of_graph(graph: Graph):
    assert len(graph) == 5


@pytest_fixture
def graph():
    nodes = [0, 1, 2, 3, 4]
    adj = {
        0: [(1, 2), (2, 1)],
        1: [(0, 2), (3, 3), (4, 1)],
        2: [(0, 1), (4, 4)],
        3: [(1, 3), (4, 1)],
        4: [(1, 1), (2, 4), (3, 1)],
    }
    computed_return_value = Graph(nodes, adj)
    return computed_return_value


@pytest_fixture
def graph_string_labels():
    nodes = ["0", "1", "2", "3", "4"]
    adj = {
        "0": [("1", 2), ("2", 1)],
        "1": [("0", 2), ("3", 3), ("4", 1)],
        "2": [("0", 1), ("4", 4)],
        "3": [("1", 3), ("4", 1)],
        "4": [("1", 1), ("2", 4), ("3", 1)],
    }
    computed_return_value = Graph(nodes, adj)
    return computed_return_value
