"""Utilities for chain chunk."""

from mage.graph_coloring_module.components.correlation_population import (
    CorrelationPopulation,
)
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


class ChainChunk(CorrelationPopulation):
    """A class that represents a population that is just a part
    of the whole population. First and last individuals of this
    population exchange information with individuals located in
    other parts of the entire population. Pieces of the population
    are ordered. The first individual communicates with the last in
    the previous piece of the population, and the last communicates
    with the first in the next piece of the population."""

    def __init__(
        self,
        graph: Graph,
        individuals: list[Individual],
        prev_indv: Individual,
        next_indv: Individual,
    ):
        super().__init__(graph, individuals)
        self.internal_prev_indv = prev_indv
        self.internal_next_indv = next_indv
        self.set_correlations()

    def get_prev_correlation_index(self, index: int) -> int:
        """Returns the index of the correlation with the
        previous individual in the chain of individuals."""
        computed_return_value = index - 1 if index - 1 >= 0 else self.size
        return computed_return_value

    def get_next_correlation_index(self, index: int) -> int:
        """Returns the index of the correlation with the
        next individual in the chain of individuals."""
        return index

    def get_prev_individual(self, index: int) -> Individual:
        """Returns the individual that precedes the
        individual on the given index."""
        if index < 0 or index >= self.size:
            raise IndexError()
        if index == 0:
            return self.internal_prev_indv
        computed_return_value = self.individuals[index - 1]
        return computed_return_value

    def get_next_individual(self, index: int) -> Individual:
        """Returns the individual that follows the
        individual on the given index."""
        if index < 0 or index >= self.size:
            raise IndexError()
        if index + 1 == self.size:
            return self.internal_next_indv
        computed_return_value = self.individuals[index + 1]
        return computed_return_value

    def set_correlations(self) -> bool:
        for i in range(self.size + 1):
            if i == self.size:
                c = self.calculate_correlation(self.individuals[0], self.internal_prev_indv)
            else:
                next_indv = self.individuals[i + 1] if i + 1 < self.size else self.internal_next_indv
                c = self.calculate_correlation(self.individuals[i], next_indv)
            self.internal_correlation.append(c)
            self.internal_cumulative_correlation += c
        return False

    def set_prev_individual(self, individual: Individual) -> bool:
        """Sets the unit that precedes the current piece of chain."""
        self.internal_cumulative_correlation -= self.internal_correlation[self.size]
        self.internal_correlation[self.size] = self.calculate_correlation(individual, self.individuals[0])
        self.internal_cumulative_correlation += self.internal_correlation[self.size]
        self.internal_prev_indv = individual
        return False

    def set_next_individual(self, individual: Individual) -> bool:
        """Sets the individual that follows the current piece of chain."""
        self.internal_cumulative_correlation -= self.internal_correlation[self.size - 1]
        self.internal_correlation[self.size - 1] = self.calculate_correlation(self.individuals[self.size - 1], individual)
        self.internal_cumulative_correlation += self.internal_correlation[self.size - 1]
        self.internal_next_indv = individual
        return False
