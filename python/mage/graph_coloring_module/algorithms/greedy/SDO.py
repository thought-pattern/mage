"""Utilities for SDO."""

from queue import PriorityQueue
from random import choice as random_choice
from random import randint as random_randint

from mage.graph_coloring_module.algorithms.algorithm import Algorithm
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.available_colors import available_colors
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate

DEFAULT_ARGUMENT_DICT = {}


class SDO(Algorithm):
    """A class that represents the SDO greedy algorithm. This algorithm sorts nodes considering
    their saturation degrees and then colors them sequentially. The saturation degree of a
    node is defined as the number of its differently colored neighbors. If it is not possible
    to uniquely determine the color, color is chosen randomly. If coloring the node with any
    possible color would cause conflicts, then the color is chosen randomly."""

    def __str__(self):
        return "SDO"

    @validate(Parameter.NO_OF_COLORS)
    def run(self, graph: Graph, parameters: dict = DEFAULT_ARGUMENT_DICT) -> Individual:
        """Returns the individual that represents the result of the SDO algorithm."""

        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        no_of_colors = param_value(graph, parameters, Parameter.NO_OF_COLORS)

        processed = [False for _ in graph.nodes]
        saturation_degrees = [0 for _ in graph.nodes]
        chromosome = [-1 for _ in graph.nodes]

        while not all(processed):
            current_node = self.get_non_processed_node(processed)
            sorted_nodes = PriorityQueue()
            sorted_nodes.put((saturation_degrees[current_node], current_node))

            while not sorted_nodes.empty():
                node = sorted_nodes.get()[1]
                if not processed[node]:
                    processed[node] = True

                    colors = available_colors(graph, no_of_colors, chromosome, node)
                    if len(colors) > 0:
                        color = random_choice(colors)
                    else:
                        color = random_randint(0, no_of_colors - 1)
                    chromosome[node] = color

                    for neigh in graph[node]:
                        if not processed[neigh]:
                            saturation_degrees[neigh] += 1
                            sorted_nodes.put((-1 * saturation_degrees[neigh], neigh))

        computed_return_value = Individual(no_of_colors, graph, chromosome)
        return computed_return_value

    def get_non_processed_node(self, processed: list[bool]) -> int:
        for i, flag in enumerate(processed):
            if not flag:
                return i
        raise RuntimeError("No unprocessed graph node remains")
