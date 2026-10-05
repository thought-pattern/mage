"""Tests for test convergence callback."""

from mage.graph_coloring_module import (
    ChainPopulation,
    ConflictError,
    ConvergenceCallback,
    Graph,
    Individual,
    Parameter,
)

# Five test nodes; the adjacency maps a node to its (neighbor, weight) pairs. Graph copies what it reads.
GRAPH_NODES = [0, 1, 2, 3, 4]
GRAPH_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2), (4, 5)],
    2: [(0, 3), (1, 2), (3, 3)],
    3: [(2, 3)],
    4: [(1, 5)],
}


def test_convergence_callback():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    indv_1 = Individual(no_of_colors=3, graph=graph, chromosome=[1, 1, 0, 2, 0], conflict_nodes={0, 1})
    indv_2 = Individual(no_of_colors=3, graph=graph, chromosome=[1, 2, 0, 0, 1])
    indv_3 = Individual(no_of_colors=3, graph=graph, chromosome=[2, 1, 0, 2, 1], conflict_nodes={1, 4})
    chain_population = ChainPopulation(graph, [indv_1, indv_2, indv_3])
    conv_callback = ConvergenceCallback()
    params = {
        Parameter.ERROR: ConflictError(),
        Parameter.CONVERGENCE_CALLBACK_TOLERANCE: 3,
        Parameter.CONVERGENCE_CALLBACK_ACTIONS: [],
    }
    conv_callback.update(graph, chain_population, params)
    assert conv_callback.internal_iteration == 1
    assert conv_callback.best_solution_error == 2

    conv_callback.update(graph, chain_population, params)
    assert conv_callback.internal_iteration == 2
    conv_callback.update(graph, chain_population, params)
    assert conv_callback.internal_iteration == 0

    conv_callback.update(graph, chain_population, params)
    assert conv_callback.internal_iteration == 1

    new_indv = Individual(no_of_colors=3, graph=graph, chromosome=[1, 2, 0, 2, 1])
    chain_population.set_individual(1, new_indv, [3])
    conv_callback.update(graph, chain_population, params)
    assert conv_callback.internal_iteration == 0
    assert conv_callback.best_solution_error == 0
