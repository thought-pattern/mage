"""Tests for test operators."""

from random import seed as random_seed

from pytest import fixture as pytest_fixture

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.operators.mutations.MIS_mutation import MISMutation
from mage.graph_coloring_module.operators.mutations.multiple_mutation import MultipleMutation
from mage.graph_coloring_module.operators.mutations.random_mutation import RandomMutation
from mage.graph_coloring_module.operators.mutations.simple_mutation import SimpleMutation
from mage.graph_coloring_module.parameters import Parameter

# Five test nodes; the adjacency maps a node to its (neighbor, weight) pairs. Graph copies what it reads.
GRAPH_NODES = [0, 1, 2, 3, 4]
GRAPH_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2), (4, 5)],
    2: [(0, 3), (1, 2), (3, 3)],
    3: [(2, 3)],
    4: [(1, 5)],
}


# Every mutation test expects the outcome of the same seeded random sequence.
@pytest_fixture(autouse=True)
def set_seed():
    random_seed(42)
    return False


def test_mis_mutation():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 0, 2, 0])
    mutated_indv, nodes = MISMutation().mutate(graph, individual)

    expected_mutated_indv_chromosome = [0, 2, 0, 2, 0]
    expected_nodes = [3, 1]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_multiple_mutation():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 0, 2, 0])
    mutated_indv, nodes = MultipleMutation().mutate(
        graph,
        individual,
        {Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES: 2, Parameter.NO_OF_COLORS: 3},
    )

    expected_mutated_indv_chromosome = [1, 1, 0, 2, 0]
    expected_nodes = [0, 0]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_random_mutation_no_conflict():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 2, 1, 0])
    mutated_indv, nodes = RandomMutation().mutate(
        graph,
        individual,
        {
            Parameter.RANDOM_MUTATION_PROBABILITY: 0.5,
            Parameter.NO_OF_COLORS: 3,
        },
    )
    expected_mutated_indv_chromosome = [0, 1, 2, 1, 0]
    expected_nodes = []
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_random_mutation():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[1, 1, 0, 2, 0], conflict_nodes={0, 1})
    mutated_indv, nodes = RandomMutation().mutate(
        graph,
        individual,
        {
            Parameter.RANDOM_MUTATION_PROBABILITY: 0.5,
            Parameter.NO_OF_COLORS: 3,
        },
    )
    expected_mutated_indv_chromosome = [2, 1, 0, 2, 0]
    expected_nodes = [0]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_simple_no_conflict():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 2, 1, 0])
    mutated_indv, nodes = SimpleMutation().mutate(graph, individual)
    expected_mutated_indv_chromosome = [0, 1, 2, 1, 0]
    expected_nodes = []
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_simple_mutation():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[1, 1, 0, 2, 0], conflict_nodes={0, 1})
    mutated_indv, nodes = SimpleMutation().mutate(graph, individual)
    expected_mutated_indv_chromosome = [0, 1, 0, 2, 0]
    expected_nodes = [0]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)
