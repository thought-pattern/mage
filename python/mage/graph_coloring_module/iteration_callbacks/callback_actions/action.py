"""Utilities for action."""

from abc import ABC, abstractmethod

from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph

DEFAULT_ARGUMENT_DICT = {}


class Action(ABC):
    """An abstract class that defines action. The action defines
    what happens when the conditions specified in the iteration
    callback are met."""

    @abstractmethod
    def execute(
        self,
        graph: Graph,
        population: Population,
        parameters: dict = DEFAULT_ARGUMENT_DICT,
    ) -> bool:
        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        ...
