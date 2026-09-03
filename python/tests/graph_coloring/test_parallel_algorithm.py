"""Behavior tests for parallel graph-coloring coordination."""

from multiprocessing import Value

from pytest import raises

from mage.graph_coloring_module import (
    ChainChunk,
    ConflictError,
    Graph,
    Individual,
    Parameter,
    QA,
)


class RecolorFirstNode:
    """Mutation that resolves the fixture's single conflicting edge."""

    def mutate(self, graph: Graph, individual: Individual, parameters: dict):
        del graph, parameters
        changed = individual.replace_unit(0, 1)
        return changed, [0]


def test_quantum_worker_publishes_early_termination():
    graph = Graph([0, 1], {0: [(1, 1)], 1: [(0, 1)]})
    conflicting = Individual(2, graph, chromosome=[0, 0])
    population = ChainChunk(graph, [conflicting], conflicting, conflicting)
    running = Value("i", 1)
    parameters = {
        Parameter.MAX_ITERATIONS: 1,
        Parameter.ERROR: ConflictError(),
        Parameter.COMMUNICATION_DALAY: 1,
        Parameter.LOGGING_DELAY: 1,
        Parameter.ITERATION_CALLBACKS: [],
        Parameter.NO_OF_PROCESSES: 1,
        Parameter.QA_TEMPERATURE: 1.0,
        Parameter.QA_MAX_STEPS: 1,
        Parameter.MUTATION: RecolorFirstNode(),
        Parameter.CONFLICT_ERR_ALPHA: 0.5,
        Parameter.CONFLICT_ERR_BETA: 0.5,
    }

    QA().algorithm(
        0,
        graph,
        population,
        {0: conflicting},
        {0: conflicting},
        {0: conflicting},
        running,
        parameters,
    )

    assert running.value == 0
    assert population.best_individual(ConflictError().individual_err).check_coloring()


def test_parallel_run_constructs_population_without_factory():
    graph = Graph([0], {0: []})
    parameters = {
        Parameter.NO_OF_PROCESSES: 2,
        Parameter.POPULATION_SIZE: 2,
        Parameter.NO_OF_COLORS: 1,
        Parameter.INIT_ALGORITHMS: [],
        Parameter.ERROR: ConflictError(),
    }

    result = QA().run(graph, parameters)

    assert isinstance(result, Individual)
    assert result.check_coloring()


def test_parallel_run_rejects_empty_population_chunks():
    graph = Graph([0], {0: []})
    parameters = {
        Parameter.NO_OF_PROCESSES: 2,
        Parameter.POPULATION_SIZE: 1,
        Parameter.NO_OF_COLORS: 1,
        Parameter.INIT_ALGORITHMS: [],
        Parameter.ERROR: ConflictError(),
    }

    with raises(ValueError, match="at least as large"):
        QA().run(graph, parameters)
