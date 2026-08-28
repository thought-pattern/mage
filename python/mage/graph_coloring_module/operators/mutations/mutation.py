"""Utilities for mutation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph

_DEFAULT_ARGUMENT_DICT = {}


class Mutation(ABC):
    """A class that represents a mutation."""

    @abstractmethod
    def mutate(
        self,
        graph: Graph,
        individual: Individual,
        parameters: Dict[str, Any] = _DEFAULT_ARGUMENT_DICT,
    ) -> Tuple[Individual, List[int]]:
        """A function that mutates the given individual and
        returns the new individual and nodes that was changed."""
        if parameters is _DEFAULT_ARGUMENT_DICT:
            parameters = _DEFAULT_ARGUMENT_DICT.copy()
        return ()
