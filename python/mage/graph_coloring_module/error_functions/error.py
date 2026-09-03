"""Utilities for error."""

from abc import ABC, abstractmethod

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph

DEFAULT_ARGUMENT_DICT = {}


class Error(ABC):
    """A class that represents an error function."""

    @abstractmethod
    def individual_err(
        self,
        graph: Graph,
        individual: Individual,
        parameters: dict = DEFAULT_ARGUMENT_DICT,
    ) -> float:
        """Calculates the error of the individual."""
        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        return 0.0

    @abstractmethod
    def population_err(
        self,
        graph: Graph,
        population: Population,
        parameters: dict = DEFAULT_ARGUMENT_DICT,
    ) -> float:
        """Calculates the population error."""
        if parameters is DEFAULT_ARGUMENT_DICT:
            parameters = DEFAULT_ARGUMENT_DICT.copy()
        return 0.0
