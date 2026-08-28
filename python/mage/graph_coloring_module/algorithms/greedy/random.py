"""Utilities for random."""

from typing import Any, Dict

from mage.graph_coloring_module.algorithms.algorithm import Algorithm
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate

_DEFAULT_ARGUMENT_DICT = {}


class Random(Algorithm):
    """A class that represents the algorithm that randomly colors nodes."""

    def __str__(self):
        return "Random"

    @validate(Parameter.NO_OF_COLORS)
    def run(
        self, graph: Graph, parameters: Dict[str, Any] = _DEFAULT_ARGUMENT_DICT
    ) -> Individual:
        if parameters is _DEFAULT_ARGUMENT_DICT:
            parameters = _DEFAULT_ARGUMENT_DICT.copy()
        no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)
        _return_value = Individual(no_of_colors, graph)
        return _return_value
