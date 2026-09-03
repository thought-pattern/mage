"""Utilities for chain population."""

from mage.graph_coloring_module.components.correlation_population import (
    CorrelationPopulation,
)
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


class ChainPopulation(CorrelationPopulation):
    """A class that represents a chain population. In this
    population, the last individual is followed by the first
    individual, and the predecessor of the first individual
    is the last individual."""

    def __init__(self, graph: Graph, individuals: list[Individual]):
        super().__init__(graph, individuals)
        self.set_correlations()

    def get_prev_correlation_index(self, index: int) -> int:
        """Returns the index of the correlation with the previous
        individual in the chain of individuals."""
        computed_return_value = index - 1 if index - 1 >= 0 else self.size - 1
        return computed_return_value

    def get_next_correlation_index(self, index: int) -> int:
        """Returns the index of the correlation with the next
        individual in the chain of individuals."""
        return index

    def get_prev_individual(self, index: int) -> Individual:
        """Returns the individual that precedes the individual on the given index."""
        if index < 0 or index >= self.size:
            raise IndexError()
        prev_ind = index - 1 if index - 1 >= 0 else self.size - 1
        computed_return_value = self.individuals[prev_ind]
        return computed_return_value

    def get_next_individual(self, index: int) -> Individual:
        """Returns the individual that follows the individual on the given index."""
        if index < 0 or index >= self.size:
            raise IndexError()
        next_ind = index + 1 if index + 1 < self.size else 0
        computed_return_value = self.individuals[next_ind]
        return computed_return_value

    def set_correlations(self) -> bool:
        for i in range(self.size):
            j = i + 1 if i + 1 < self.size else 0
            c = self.calculate_correlation(self.individuals[i], self.individuals[j])
            self.internal_correlation.append(c)
            self.internal_cumulative_correlation += c
        return False
