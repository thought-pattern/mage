"""Utilities for population."""

from abc import ABC, abstractmethod

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


class Population(ABC):
    """An abstract class that represents a population. A population
    contains individuals that are placed in a chain and exchange
    information with individuals that are located next to it."""

    def __init__(self, graph: Graph, individuals: list[Individual]):
        self.internal_size = len(individuals)
        self.internal_individuals = individuals
        self.internal_best_individuals = self.internal_individuals[:]
        self.internal_graph = graph

        self.internal_sum_conflicts_weight = 0
        self.calculate_metrics()

    def __len__(self) -> int:
        """Returns size of the population."""
        return self.internal_size

    def __getitem__(self, index: int) -> Individual:
        """Returns an individual that is placed on the given index."""
        computed_return_value = self.internal_individuals[index]
        return computed_return_value

    @abstractmethod
    def get_prev_individual(self, index: int) -> Individual:
        """Returns the individual that precedes the individual
        on the given index in the chain of individuals."""
        ...

    @abstractmethod
    def get_next_individual(self, index: int) -> Individual:
        """Returns the individual that follows the individual
        on the given index in the chain of individuals."""
        ...

    @property
    def individuals(self) -> list[Individual]:
        """Returns a list of individuals."""
        return self.internal_individuals

    @property
    def best_individuals(self) -> list[Individual]:
        """Returns a list of individuals that had
        the smallest error through iterations."""
        return self.internal_best_individuals

    @property
    def size(self) -> int:
        """Returns the size of the population."""
        return self.internal_size

    @property
    def mean_conflicts_weight(self) -> float:
        """Returns the average sum of weights of conflicting edges
        in individuals contained in population."""
        computed_return_value = self.internal_sum_conflicts_weight / self.size
        return computed_return_value

    @property
    def sum_conflicts_weight(self) -> float:
        """Returns the sum of sum of weights of conflicting edges
        in individuals contained in population"""
        return self.internal_sum_conflicts_weight

    def set_individual(self, index: int, individual: Individual, diff_nodes: list[int]) -> int:
        """Sets the individual on the specified index to the given individual
        and updates appropriate correlations and metrics."""
        old_individual = self.internal_individuals[index]
        self.internal_individuals[index] = individual
        self.update_metrics(index, old_individual)
        return 0

    def best_individual_index(self, error_function) -> int:
        """Returns the index of the individual with the smallest error."""
        errors = self.individuals_errors(error_function)
        computed_return_value = min(range(len(errors)), key=errors.__getitem__)
        return computed_return_value

    def worst_individual_index(self, error_function) -> int:
        """Returns the index of the individual with the largest error."""
        errors = self.individuals_errors(error_function)
        computed_return_value = max(range(len(errors)), key=errors.__getitem__)
        return computed_return_value

    def best_individual(self, error_function) -> Individual:
        """Returns the individual with the smallest error."""
        computed_return_value = self.internal_individuals[self.best_individual_index(error_function)]
        return computed_return_value

    def worst_individual(self, error_function) -> Individual:
        """Returns the individual with the largest error."""
        computed_return_value = self.internal_individuals[self.worst_individual_index(error_function)]
        return computed_return_value

    def individuals_errors(self, error_function) -> list[float]:
        """Returns a list of individuals errors."""
        if not callable(error_function):
            raise TypeError("error_function must be callable")
        computed_return_value = []
        for individual in self.individuals:
            error_value = error_function(self.internal_graph, individual)
            if isinstance(error_value, bool) or not isinstance(error_value, (int, float)):
                raise TypeError("error_function must return a number")
            computed_return_value.append(float(error_value))
        return computed_return_value

    def min_error(self, error_function) -> float:
        """Returns the smallest error in the population."""
        computed_return_value = min(self.individuals_errors(error_function))
        return computed_return_value

    def max_error(self, error_function) -> float:
        """Returns the largest error in the population."""
        computed_return_value = max(self.individuals_errors(error_function))
        return computed_return_value

    def calculate_metrics(self) -> bool:
        for individual in self.individuals:
            self.internal_sum_conflicts_weight += individual.conflicts_weight
        return False

    def update_metrics(self, ind: int, old_indv: Individual) -> bool:
        new_indv = self.individuals[ind]
        self.internal_sum_conflicts_weight -= old_indv.conflicts_weight
        self.internal_sum_conflicts_weight += new_indv.conflicts_weight

        best_conflicts_weight = self.internal_best_individuals[ind].conflicts_weight
        new_conflicts_weight = new_indv.conflicts_weight
        if new_conflicts_weight < best_conflicts_weight:
            self.internal_best_individuals[ind] = new_indv
        return False
