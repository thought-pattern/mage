"""Utilities for parallel algorithm."""

from abc import ABC, abstractmethod
from logging import getLogger as logging_getLogger
from multiprocessing import Manager as mp_Manager
from multiprocessing import Process as mp_Process
from multiprocessing import Value as mp_Value

from mage.graph_coloring_module.algorithms.algorithm import Algorithm
from mage.graph_coloring_module.components.chain_chunk import ChainChunk
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.exceptions import PopulationCreationException
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate

logger = logging_getLogger("graph_coloring")


class ParallelAlgorithm(Algorithm, ABC):
    """A class that represents an abstract parallel algorithm."""

    @validate(
        Parameter.NO_OF_PROCESSES,
        Parameter.POPULATION_SIZE,
        Parameter.NO_OF_COLORS,
        Parameter.INIT_ALGORITHMS,
        Parameter.ERROR,
    )
    def run(self, graph: Graph, parameters: dict) -> Individual:
        """Runs the algorithm in a given number of processes and returns the best individual.

        Parameters that must be specified:
        :no_of_processes: the number of processes to run an algorithm in
        :error: a function that defines an error"""

        no_of_processes = param_value(graph, parameters, Parameter.NO_OF_PROCESSES)
        population_size = param_value(graph, parameters, Parameter.POPULATION_SIZE)
        no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)
        init_algorithms = param_value(graph, parameters, Parameter.INIT_ALGORITHMS)
        error = param_value(graph, parameters, Parameter.ERROR)
        if not isinstance(no_of_processes, int) or isinstance(no_of_processes, bool) or no_of_processes < 1:
            raise ValueError("no_of_processes must be a positive integer")
        if not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < no_of_processes:
            raise ValueError("population_size must be an integer at least as large as no_of_processes")
        if not isinstance(no_of_colors, int) or isinstance(no_of_colors, bool) or no_of_colors < 1:
            raise ValueError("no_of_colors must be a positive integer")
        if not isinstance(init_algorithms, list):
            raise TypeError("init_algorithms must be a list")
        individual_error = getattr(error, "individual_err", False)
        if not callable(individual_error):
            raise TypeError("error must provide individual_err()")

        def individual_error_value(error_graph: Graph, individual: Individual) -> float:
            value = individual_error(error_graph, individual, parameters)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("error.individual_err() must return a number")
            numeric_value = float(value)
            return numeric_value

        individuals = []
        for algorithm in init_algorithms[:population_size]:
            run_algorithm = getattr(algorithm, "run", False)
            if not callable(run_algorithm):
                raise TypeError("every initialization algorithm must provide run()")
            individual = run_algorithm(graph, parameters)
            if not isinstance(individual, Individual):
                raise PopulationCreationException("An initialization algorithm did not produce an Individual")
            individuals.append(individual)
        individuals.extend(Individual(no_of_colors, graph) for _ in range(population_size - len(individuals)))
        chunk_size, remainder = divmod(population_size, no_of_processes)
        chunks = [
            list(individuals[pid * chunk_size + min(pid, remainder) : (pid + 1) * chunk_size + min(pid + 1, remainder)])
            for pid in range(no_of_processes)
        ]
        populations = [
            ChainChunk(
                graph,
                chunk,
                chunks[pid - 1][-1],
                chunks[(pid + 1) % no_of_processes][0],
            )
            for pid, chunk in enumerate(chunks)
        ]

        initial_best_solutions = {
            pid: population.best_individual(individual_error_value) for pid, population in enumerate(populations)
        }
        if any(individual_error_value(graph, individual) < 1e-5 for individual in initial_best_solutions.values()):
            best_individual = min(
                initial_best_solutions.values(),
                key=lambda individual: individual_error_value(graph, individual),
            )
            return best_individual

        try:
            manager_context = mp_Manager()
        except Exception as error_value:
            raise RuntimeError("Failed to create graph-coloring process coordination") from error_value

        with manager_context as manager:
            running_flag = mp_Value("i", 1)
            best_solutions = manager.dict(initial_best_solutions)
            last_individuals = manager.dict()
            first_individuals = manager.dict()

            for pid in range(no_of_processes):
                last_individuals[pid] = populations[pid][-1]
                first_individuals[pid] = populations[pid][0]

            processes = [
                mp_Process(
                    target=self.algorithm,
                    args=(
                        pid,
                        graph,
                        populations[pid],
                        best_solutions,
                        first_individuals,
                        last_individuals,
                        running_flag,
                        parameters,
                    ),
                )
                for pid in range(no_of_processes)
            ]

            started_processes = []
            try:
                for process in processes:
                    process.start()
                    started_processes.append(process)
            except Exception as error_value:
                for process in started_processes:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                raise RuntimeError("Failed to start every graph-coloring worker") from error_value

            for process in started_processes:
                process.join()

            failed_processes = [process.pid for process in started_processes if process.exitcode != 0]
            if failed_processes:
                raise RuntimeError(f"Graph-coloring workers failed: {failed_processes}")

            best_individual = min(
                best_solutions.values(),
                key=lambda individual: individual_error_value(graph, individual),
            )

            return best_individual

    @abstractmethod
    def algorithm(
        self,
        pid: int,
        graph: Graph,
        population: Population,
        best_solutions: dict[int, Individual],
        first_individuals: dict[int, Individual],
        last_individuals: dict[int, Individual],
        running_flag,
        parameters: dict,
    ) -> bool:
        """A function that executes an algorithm."""
        ...

    def previous_pid(self, pid: int, no_of_processes: int):
        prev_pid = pid - 1 if pid > 0 else no_of_processes - 1
        return prev_pid

    def internal_next_pid(self, pid: int, no_of_processes: int):
        next_pid = pid + 1 if pid + 1 < no_of_processes else 0
        return next_pid
