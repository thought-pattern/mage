"""Tests for test conflict error."""

from math import fabs as math_fabs
from random import seed as random_seed

from pytest import fixture as pytest_fixture, mark as pytest_mark

from mage.graph_coloring_module import (
    ChainPopulation,
    ConflictError,
    Graph,
    Individual,
    Parameter,
)

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
# Individual keyword arguments of the population members; Individual copies them.
CHAIN_MEMBERS = (
    {"chromosome": [1, 1, 0, 2, 0], "conflict_nodes": {0, 1}},
    {"chromosome": [1, 2, 0, 2, 1]},
    {"chromosome": [2, 1, 0, 2, 1], "conflict_nodes": {1, 4}},
)


@pytest_fixture
def set_seed():
    random_seed(42)
    return False


@pytest_mark.parametrize(
    "adjacency, no_of_colors, chromosome, expected_error",
    [
        (GRAPH_ADJACENCY, 3, [1, 1, 0, 2, 0], 2),
        (GRAPH_ADJACENCY, 3, [1, 1, 2, 2, 0], 5),
        (NOT_CONNECTED_ADJACENCY, 3, [1, 1, 0, 2, 0], 2),
    ],
)
def test_individual_error_no_setting(set_seed, adjacency, no_of_colors, chromosome, expected_error):
    graph = Graph(GRAPH_NODES, adjacency)
    individual = Individual(no_of_colors=no_of_colors, graph=graph, chromosome=chromosome)
    error = ConflictError().individual_err(graph, individual)
    assert error == expected_error


def test_population_error(set_seed):
    population_graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=population_graph, **member) for member in CHAIN_MEMBERS]
    chain_population = ChainPopulation(population_graph, members)
    error = ConflictError().population_err(
        Graph(GRAPH_NODES, GRAPH_ADJACENCY),
        chain_population,
        {Parameter.CONFLICT_ERR_ALPHA: 0.5, Parameter.CONFLICT_ERR_BETA: 0.5},
    )

    expected_error = 6.5
    assert math_fabs(error - expected_error) < 1e-5


def test_delta_error_function():
    g = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    error = ConflictError().delta(
        g,
        Individual(no_of_colors=3, graph=g, chromosome=[1, 2, 0, 2, 1]),
        Individual(no_of_colors=3, graph=g, chromosome=[1, 2, 2, 2, 1]),
        -8,
        {Parameter.CONFLICT_ERR_ALPHA: 0.5, Parameter.CONFLICT_ERR_BETA: 0.5},
    )

    expected_error = -1.5
    assert math_fabs(error - expected_error) < 1e-5
