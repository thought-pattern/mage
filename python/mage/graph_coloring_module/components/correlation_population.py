"""Utilities for correlation population."""

from abc import abstractmethod

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph


class CorrelationPopulation(Population):
    def __init__(self, graph: Graph, individuals: list[Individual]):
        super().__init__(graph, individuals)
        self.internal_cumulative_correlation = 0
        self.internal_correlation = []

    @abstractmethod
    def set_correlations(self) -> bool:
        """Calculates the correlations between individuals
        and stores them in correlation list."""
        ...

    @abstractmethod
    def get_prev_correlation_index(self, index: int) -> int:
        """Returns the index of the correlation between an individual
        on the given index and the previous individual in the chain of individuals."""
        return 0

    @abstractmethod
    def get_next_correlation_index(self, index: int) -> int:
        """Returns the index of the correlation between an individual
        on the given index and the next individual in the chain of individuals."""
        return 0

    def set_individual(self, index: int, individual: Individual, diff_nodes: list[int]) -> bool:
        """Sets the individual on the specified index to the given individual
        and updates appropriate correlations and metrics."""
        old_individual = self.internal_individuals[index]
        self.internal_individuals[index] = individual
        self.update_correlation(index, old_individual, diff_nodes)
        self.update_metrics(index, old_individual)
        return False

    @property
    def correlation(self) -> list[float]:
        """Returns a list that contains correlations between individuals.
        Correlation on the index i is the correlation between the individual
        placed on the index i in the list of individuals and the individual
        that is next to that individual."""
        return self.internal_correlation

    @property
    def cumulative_correlation(self) -> float:
        """Returns the cumulative correlation of the population."""
        return self.internal_cumulative_correlation

    def correlations(self, index: int) -> tuple[int, int]:
        """Returns correlations between a given individual
        and the previous and next individual."""
        prev_index = self.get_prev_correlation_index(index)
        next_index = self.get_next_correlation_index(index)
        computed_return_value = self.internal_correlation[prev_index], self.internal_correlation[next_index]
        return computed_return_value

    def calculate_correlation(self, first: Individual, second: Individual) -> float:
        correlation = 0
        for node_1 in self.internal_graph.nodes:
            for node_2 in range(node_1 + 1, len(self.internal_graph)):
                S_first = -1 if first[node_1] == first[node_2] else 1
                S_second = -1 if second[node_1] == second[node_2] else 1
                correlation += S_first * S_second
        return correlation

    def update_correlation(self, index: int, old_individual: Individual, nodes: list[int]) -> int:
        next_correlation_index = self.get_next_correlation_index(index)
        prev_correlation_index = self.get_prev_correlation_index(index)

        new_individual = self.individuals[index]
        prev_individual = self.get_prev_individual(index)
        next_individual = self.get_next_individual(index)

        correlation_prev_delta = 0
        correlation_next_delta = 0
        processed = [False for _ in range(old_individual.no_of_units)]

        for node in nodes:
            for neigh in self.internal_graph[node]:
                if not processed[neigh]:
                    S_old = -1 if old_individual[node] == old_individual[neigh] else 1
                    S_new = -1 if new_individual[node] == new_individual[neigh] else 1
                    S_prev = -1 if prev_individual[node] == prev_individual[neigh] else 1
                    S_next = -1 if next_individual[node] == next_individual[neigh] else 1
                    correlation_prev_delta += (S_new * S_prev) - (S_old * S_prev)
                    correlation_next_delta += (S_new * S_next) - (S_old * S_next)
            processed[node] = True

        self.internal_correlation[prev_correlation_index] += correlation_prev_delta
        self.internal_correlation[next_correlation_index] += correlation_next_delta
        delta_corr = correlation_prev_delta + correlation_next_delta
        self.internal_cumulative_correlation += delta_corr
        return delta_corr
