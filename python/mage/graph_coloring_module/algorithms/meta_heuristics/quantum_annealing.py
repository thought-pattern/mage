"""Utilities for quantum annealing."""

from logging import getLogger as logging_getLogger
from math import exp as math_exp
from math import fabs as math_fabs
from random import random as random_random

from mage.graph_coloring_module.algorithms.meta_heuristics.parallel_algorithm import (
    ParallelAlgorithm,
)
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.chain_chunk import ChainChunk
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate

logger = logging_getLogger("graph_coloring")


class QA(ParallelAlgorithm):
    """A class that represents the quantum annealing algorithm."""

    def __str__(self):
        return "QA"

    @validate(
        Parameter.MAX_ITERATIONS,
        Parameter.ERROR,
        Parameter.COMMUNICATION_DALAY,
        Parameter.LOGGING_DELAY,
        Parameter.ITERATION_CALLBACKS,
        Parameter.NO_OF_PROCESSES,
    )
    def algorithm(
        self,
        pid: int,
        graph: Graph,
        population: ChainChunk,
        best_solutions: dict[int, Individual],
        first_individuals: dict[int, Individual],
        last_individuals: dict[int, Individual],
        running_flag,
        parameters: dict,
    ) -> bool:
        """Function that executes the QA algorithm. The resulting population
        is written to the queue named results."""

        max_iterations = param_value(graph, parameters, Parameter.MAX_ITERATIONS)
        error = param_value(graph, parameters, Parameter.ERROR)
        communication_delay = param_value(graph, parameters, Parameter.COMMUNICATION_DALAY)
        logging_delay = param_value(graph, parameters, Parameter.LOGGING_DELAY)
        iteration_callbacks = param_value(graph, parameters, Parameter.ITERATION_CALLBACKS)
        no_of_processes = param_value(graph, parameters, Parameter.NO_OF_PROCESSES)

        for iteration in range(max_iterations):
            if running_flag.value == 0:
                return False

            for i in range(len(population)):
                self.markow_chain(graph, population, i, parameters)

            best_individual = population.best_individual(error.individual_err)
            if error.individual_err(graph, best_individual, parameters) < error.individual_err(
                graph, best_solutions.get(pid, False), parameters
            ):
                best_solutions[pid] = best_individual

            if math_fabs(error.individual_err(graph, best_individual, parameters)) < 1e-5:
                with running_flag.get_lock():
                    running_flag.value = 0
                return False

            if iteration % communication_delay == 0:
                first_individuals[pid] = population[0]
                last_individuals[pid] = population[-1]

                next_individual = first_individuals.get(self.internal_next_pid(pid, no_of_processes), False)
                previous_individual = last_individuals.get(self.previous_pid(pid, no_of_processes), False)
                if not isinstance(next_individual, Individual) or not isinstance(previous_individual, Individual):
                    raise RuntimeError("Parallel chain neighbors were not initialized")
                population.set_next_individual(next_individual)
                population.set_prev_individual(previous_individual)

            for callback in iteration_callbacks:
                callback.update(graph, population, parameters)

            if iteration % logging_delay == 0:
                logger.info("Id: {} Iteration: {} Error: {}".format(pid, iteration, population.min_error(error.individual_err)))

        logger.info("Id: {} Iteration: {} Error: {}".format(pid, iteration, population.min_error(error.individual_err)))

        for callback in iteration_callbacks:
            callback.end(graph, population, parameters)

        return False

    @validate(
        Parameter.QA_TEMPERATURE,
        Parameter.QA_MAX_STEPS,
        Parameter.MUTATION,
        Parameter.ERROR,
    )
    def markow_chain(
        self,
        graph: Graph,
        population: Population,
        index: int,
        parameters: dict,
    ) -> bool:
        temperature = param_value(graph, parameters, Parameter.QA_TEMPERATURE)
        max_steps = param_value(graph, parameters, Parameter.QA_MAX_STEPS)
        mutation = param_value(graph, parameters, Parameter.MUTATION)
        error = param_value(graph, parameters, Parameter.ERROR)

        for _ in range(max_steps):
            individual = population[index]
            population_error_old = error.population_err(graph, population, parameters)
            new_individual, diff_nodes = mutation.mutate(graph, individual, parameters)
            delta_individual_error = error.individual_err(graph, new_individual) - error.individual_err(graph, individual)
            population.set_individual(index, new_individual, diff_nodes)
            population_error_new = error.population_err(graph, population, parameters)
            delta_population_error = population_error_new - population_error_old

            if delta_individual_error > 0 or delta_population_error > 0:
                try:
                    probability = 1 - math_exp((-1 * delta_population_error) / temperature)
                except OverflowError:
                    probability = 1
                if random_random() <= probability:
                    population.set_individual(index, individual, diff_nodes)
        return False
