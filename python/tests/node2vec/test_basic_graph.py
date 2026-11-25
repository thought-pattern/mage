"""Tests for test basic graph."""

from pytest import mark as pytest_mark

from mage.node2vec.graph import GraphHolder

DIRECT_GRAPH_EDGES_WEIGHTS = {
    (0, 1): 0.5,
    (0, 2): 1,
    (0, 4): 1,
    (1, 5): 1,
    (1, 6): 1,
    (2, 5): 1,
    (3, 0): 1,
    (4, 6): 1,
    (7, 1): 1,
    (7, 5): 1,
    (7, 6): 1,
}


DIRECT_GRAPH_NODE_NEIGHBORS = {
    0: [1, 2, 4],
    1: [5, 6],
    2: [5],
    3: [0],
    4: [6],
    5: [],
    6: [],
    7: [1, 5, 6],
}

UNDIRECT_GRAPH_NODE_NEIGHBORS = {
    0: [1, 2, 3, 4],
    1: [0, 5, 6, 7],
    2: [0, 5],
    3: [0],
    4: [0, 6],
    5: [1, 2, 7],
    6: [1, 4, 7],
    7: [1, 5, 6],
}


@pytest_mark.parametrize("is_directed", [True, False])
def test_graph_edges_from_dict(is_directed):
    # GraphHolder reads the edge weights without mutating them.
    basic_graph_from_dict = GraphHolder(DIRECT_GRAPH_EDGES_WEIGHTS, is_directed)
    graph_edges = basic_graph_from_dict.get_edges()
    if basic_graph_from_dict.is_directed:
        assert len(graph_edges) == len(DIRECT_GRAPH_EDGES_WEIGHTS)
    else:
        assert len(graph_edges) == len(DIRECT_GRAPH_EDGES_WEIGHTS) * 2

    for edge in DIRECT_GRAPH_EDGES_WEIGHTS:
        assert basic_graph_from_dict.has_edge(edge[0], edge[1]) is True
        assert basic_graph_from_dict.has_edge(edge[1], edge[0]) is not basic_graph_from_dict.is_directed

    expected_neighbors = DIRECT_GRAPH_NODE_NEIGHBORS if basic_graph_from_dict.is_directed else UNDIRECT_GRAPH_NODE_NEIGHBORS
    for node in basic_graph_from_dict.nodes:
        assert node in expected_neighbors
        assert basic_graph_from_dict.get_neighbors(node) == expected_neighbors.get(node, [])

    if not basic_graph_from_dict.is_directed:
        assert basic_graph_from_dict.get_edge_weight(0, 1) == 0.5
        assert basic_graph_from_dict.get_edge_weight(1, 0) == 0.5
