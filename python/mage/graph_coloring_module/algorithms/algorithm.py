"""Utilities for algorithm."""

from abc import ABC, abstractmethod

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


class Algorithm(ABC):
    """An abstract class that represents an algorithm."""

    @abstractmethod
    def run(self, graph: Graph, parameters: dict) -> Individual:
        """Runs the algorithm and returns the best individual."""
        ...
