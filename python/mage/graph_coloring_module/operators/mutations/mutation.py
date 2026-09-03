"""Utilities for mutation."""

from abc import ABC, abstractmethod

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph

DEFAULT_ARGUMENT_DICT = {}


class Mutation(ABC):
    """A class that represents a mutation."""

    @abstractmethod
    def mutate(
        self,
        graph: Graph,
        individual: Individual,
        parameters: dict = DEFAULT_ARGUMENT_DICT,
    ) -> tuple[Individual, list[int]]:
        """A function that mutates the given individual and
        returns the new individual and nodes that was changed."""
        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        raise NotImplementedError
