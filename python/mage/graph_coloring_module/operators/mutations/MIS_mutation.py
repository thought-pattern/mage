"""Utilities for MIS mutation."""

from random import shuffle as random_shuffle

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.operators.mutations.mutation import Mutation

DEFAULT_ARGUMENT_DICT = {}


class MISMutation(Mutation):
    """A class that represents the maximal independent set mutation.
    This mutation finds one maximal independent set and changes
    colors of all nodes in the set to the same color."""

    def __str__(self):
        return "MISMutation"

    def mutate(
        self,
        graph: Graph,
        individual: Individual,
        parameters: dict = DEFAULT_ARGUMENT_DICT,
    ) -> tuple[Individual, list[int]]:
        """A function that mutates the given individual and
        returns the new individual and nodes that were changed."""

        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        maximal_independent_set = self.INTERNAL_MIS(graph)
        if len(maximal_independent_set) > 0:
            color = individual[maximal_independent_set[0]]
            colors = [color for _ in range(len(maximal_independent_set))]
            mutated_individual = individual.replace_units(maximal_independent_set, colors)
            return mutated_individual, maximal_independent_set
        return individual, []

    def INTERNAL_MIS(self, graph: Graph) -> list[int]:
        """A function that finds the maximal independent set. The first step
        is to shuffle nodes and add the first node to the maximal independent set.
        After that, all those nodes that do not have neighbors in the MIS are
        sequentially added to the MIS. ."""

        nodes = list(graph.nodes)
        random_shuffle(nodes)
        mis_flags = [False for _ in range(len(graph))]
        maximal_independent_set = []

        for node in nodes:
            include = True
            for neigh in graph[node]:
                if mis_flags[neigh]:
                    include = False
                    break
            if include:
                maximal_independent_set.append(node)
                mis_flags[node] = True

        return maximal_independent_set
