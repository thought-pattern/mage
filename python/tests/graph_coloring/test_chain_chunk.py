"""Tests for test chain chunk."""

from pytest import raises as pytest_raises

from mage.graph_coloring_module import ChainChunk, Graph, Individual

# Five test nodes; the adjacency maps a node to its (neighbor, weight) pairs. Graph copies what it reads.
GRAPH_NODES = [0, 1, 2, 3, 4]
GRAPH_ADJACENCY = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 2), (4, 5)],
    2: [(0, 3), (1, 2), (3, 3)],
    3: [(2, 3)],
    4: [(1, 5)],
}
# Individual keyword arguments of the chunk members and of its neighbors before and after it; Individual copies them.
CHUNK_MEMBERS = (
    {"chromosome": [1, 1, 0, 2, 0], "conflict_nodes": {0, 1}},
    {"chromosome": [1, 2, 0, 2, 1]},
    {"chromosome": [2, 1, 0, 2, 1], "conflict_nodes": {1, 4}},
)
PREV_MEMBER = {"chromosome": [2, 1, 0, 0, 0], "conflict_nodes": {2, 3}}
NEXT_MEMBER = {"chromosome": [0, 1, 2, 1, 0]}


def test_get_prev_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    result_indv = chain_chunk_population.get_prev_individual(2)
    expected_indv = chain_chunk_population[1]

    assert result_indv == expected_indv


def test_prev_out_of_range():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    with pytest_raises(IndexError):
        chain_chunk_population.get_prev_individual(10)


def test_prev_individual_negative_index():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    with pytest_raises(IndexError):
        chain_chunk_population.get_prev_individual(-2)


def test_prev_individual_of_first_item():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    result_indv = chain_chunk_population.get_prev_individual(0)
    expected_indv = chain_chunk_population.internal_prev_indv

    assert result_indv == expected_indv


def test_get_next_individual():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    result_indv = chain_chunk_population.get_next_individual(1)
    expected_indv = chain_chunk_population[2]

    assert result_indv == expected_indv


def test_next_individual_last_item():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    result_indv = chain_chunk_population.get_next_individual(2)
    expected_indv = chain_chunk_population.internal_next_indv

    assert result_indv == expected_indv


def test_next_out_of_range():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    with pytest_raises(IndexError):
        chain_chunk_population.get_next_individual(10)


def test_next_negative_index():
    graph = Graph(GRAPH_NODES, GRAPH_ADJACENCY)
    members = [Individual(no_of_colors=3, graph=graph, **member) for member in CHUNK_MEMBERS]
    prev_indv = Individual(no_of_colors=3, graph=graph, **PREV_MEMBER)
    next_indv = Individual(no_of_colors=3, graph=graph, **NEXT_MEMBER)
    chain_chunk_population = ChainChunk(graph, members, prev_indv, next_indv)
    with pytest_raises(IndexError):
        chain_chunk_population.get_next_individual(-1)
