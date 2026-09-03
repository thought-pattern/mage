"""Utilities for matplotlib callback."""

from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.iteration_callbacks.iteration_callback import (
    IterationCallback,
)


class MatplotlibCallback(IterationCallback):
    def __init__(self):
        super().__init__()

    def update(self, graph: Graph, population: Population, parameters: dict):
        return False

    def end(self, graph: Graph, population: Population, parameters: dict):
        return False
