"""Utilities for generate individuals."""

from logging import getLogger as logging_getLogger
from typing import Any, Dict, List

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.exceptions import PopulationCreationException
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate

_DEFAULT_ARGUMENT_DICT = {}

logger = logging_getLogger("graph_coloring")


@validate(Parameter.POPULATION_SIZE, Parameter.NO_OF_COLORS)
def generate_individuals(
    graph: Graph, parameters: Dict[str, Any] = _DEFAULT_ARGUMENT_DICT
) -> List[Individual]:
    """A function that creates a list of individuals in which some individuals
    are the results of the given algorithms. If more algorithms are given than
    the population size, then the remainder is ignored. If population creation has failed,
    an exception is raised."""

    if parameters is _DEFAULT_ARGUMENT_DICT:
        parameters = _DEFAULT_ARGUMENT_DICT.copy()
    population_size = param_value(graph, parameters, Parameter.POPULATION_SIZE)
    no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)
    algorithms = param_value(graph, parameters, Parameter.INIT_ALGORITHMS)

    individuals = []
    if algorithms is not False:
        for algorithm in algorithms:
            individual = algorithm.run(graph, parameters)
            if individual is False:
                logger.error("Population creation has not succeeded.")
                raise PopulationCreationException(
                    "Population creation has not succeeded."
                )
            if len(individuals) < population_size:
                individuals.append(individual)

    individuals.extend(
        [
            Individual(no_of_colors, graph)
            for _ in range(population_size - len(individuals))
        ]
    )
    return individuals
