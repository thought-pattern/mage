"""Tests for test chain population."""

from pytest import raises as pytest_raises

from mage.graph_coloring_module import ChainPopulation, Graph, Individual

# Five test nodes; the adjacency maps a node to its (neighbor, weight) pairs. Graph copies what it reads.
GRAPH_NODES = [0, 1, 2, 3, 4]
GRAPH_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2), (4, 5)],
    2: [(0, 3), (1, 2), (3, 3)],
    3: [(2, 3)],
    4: [(1, 5)],
}
# Individual keyword arguments of the population members; Individual copies them.
CHAIN_MEMBERS = (
    {"chromosome": [1, 1, 0, 2, 0], "conflict_nodes": {0, 1}},
    {"chromosome": [1, 2, 0, 2, 1]},
    {"chromosome": [2, 1, 0, 2, 1], "conflict_nodes": {1, 4}},
)


def test_previous_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.get_prev_individual(2)
    expected_indv = chain_population[1]

    assert result_indv == expected_indv


def test_previous_negative_index():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    with pytest_raises(IndexError):
        chain_population.get_prev_individual(-2)


def test_previous_out_of_range():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    with pytest_raises(IndexError):
        chain_population.get_prev_individual(10)


def test_previous_first_item():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.get_prev_individual(0)
    expected_indv = chain_population[2]

    assert result_indv == expected_indv


def test_next_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.get_next_individual(0)
    expected_indv = chain_population[1]

    assert result_indv == expected_indv


def test_next_individual_last_item():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.get_next_individual(2)
    expected_indv = chain_population[0]

    assert result_indv == expected_indv


def test_next_negative_index():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    with pytest_raises(IndexError):
        chain_population.get_next_individual(-2)


def test_next_out_of_range():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.get_next_individual(2)
    expected_indv = chain_population[0]

    assert result_indv == expected_indv
