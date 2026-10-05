"""Tests for test population."""

from math import fabs as math_fabs

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


def error_func(graph, individual):
    return individual.conflicts_weight


def test_chain_population_best_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.best_individual(error_func)
    expected_indv = chain_population[1]
    assert result_indv == expected_indv


def test_chain_population_worst_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.worst_individual(error_func)
    expected_indv = chain_population[2]
    assert result_indv == expected_indv


def test_chain_population_min_error():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.min_error(error_func)
    expected_indv = 0
    assert math_fabs(result_indv - expected_indv) < 1e-5


def test_chain_population_max_error():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.max_error(error_func)
    expected_indv = 5
    assert math_fabs(result_indv - expected_indv) < 1e-5


def test_chain_population_mean_conflicts_weights():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.mean_conflicts_weight
    expected_indv = 7 / 3
    assert math_fabs(result_indv - expected_indv) < 1e-5


def test_chain_population_sum_conflicts_weight():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result_indv = chain_population.sum_conflicts_weight
    expected_indv = 7
    assert math_fabs(result_indv - expected_indv) < 1e-5


def test_chain_population_correlation():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    result = chain_population.correlation
    expected = [2, 2, 2]
    assert expected == result


def test_set_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    chain_population = ChainPopulation(graph, [Individual(no_of_colors=3, graph=graph, **member) for member in CHAIN_MEMBERS])
    new_indv = Individual(no_of_colors=3, graph=graph, chromosome=[1, 2, 2, 2, 1])
    chain_population.set_individual(1, new_indv, [2])
    assert chain_population.correlation == [-2, -2, 2]
    assert chain_population.sum_conflicts_weight == 12
    assert chain_population.cumulative_correlation == -2
    assert chain_population.mean_conflicts_weight == 4
