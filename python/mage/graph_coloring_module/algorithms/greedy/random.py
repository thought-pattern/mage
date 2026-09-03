"""Utilities for random."""

from mage.graph_coloring_module.algorithms.algorithm import Algorithm
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate

DEFAULT_ARGUMENT_DICT = {}


class Random(Algorithm):
    """A class that represents the algorithm that randomly colors nodes."""

    def __str__(self):
        return "Random"

    @validate(Parameter.NO_OF_COLORS)
    def run(self, graph: Graph, parameters: dict[object, object] = DEFAULT_ARGUMENT_DICT) -> Individual:
        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)
        if not isinstance(no_of_colors, int) or isinstance(no_of_colors, bool) or no_of_colors <= 0:
            raise ValueError("no_of_colors must be a positive integer")
        computed_return_value = Individual(no_of_colors, graph)
        return computed_return_value
