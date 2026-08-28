"""Utilities for population."""

from abc import ABC, abstractmethod
from typing import Callable, List

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


class Population(ABC):
    """An abstract class that represents a population. A population
    contains individuals that are placed in a chain and exchange
    information with individuals that are located next to it."""

    def __init__(self, graph: Graph, individuals: List[Individual]):
        self._size = len(individuals)
        self._individuals = individuals
        self._best_individuals = self._individuals[:]
        self._graph = graph

        self._sum_conflicts_weight = 0
        self._calculate_metrics()

    def __len__(self) -> int:
        """Returns size of the population."""
        return self._size

    def __getitem__(self, index: int) -> Individual:
        """Returns an individual that is placed on the given index."""
        _return_value = self._individuals[index]
        return _return_value

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
    def individuals(self) -> List[Individual]:
        """Returns a list of individuals."""
        return self._individuals

    @property
    def best_individuals(self) -> List[Individual]:
        """Returns a list of individuals that had
        the smallest error through iterations."""
        return self._best_individuals

    @property
    def size(self) -> int:
        """Returns the size of the population."""
        return self._size

    @property
    def mean_conflicts_weight(self) -> float:
        """Returns the average sum of weights of conflicting edges
        in individuals contained in population."""
        _return_value = self._sum_conflicts_weight / self.size
        return _return_value

    @property
    def sum_conflicts_weight(self) -> float:
        """Returns the sum of sum of weights of conflicting edges
        in individuals contained in population"""
        return self._sum_conflicts_weight

    def set_individual(
        self, index: int, individual: Individual, diff_nodes: List[int]
    ) -> int:
        """Sets the individual on the specified index to the given individual
        and updates appropriate correlations and metrics."""
        old_individual = self._individuals[index]
        self._individuals[index] = individual
        self._update_metrics(index, old_individual)
        return 0

    def best_individual_index(
        self, error_function: Callable[[Graph, Individual], float]
    ) -> int:
        """Returns the index of the individual with the smallest error."""
        errors = self.individuals_errors(error_function)
        _return_value = min(range(len(errors)), key=errors.__getitem__)
        return _return_value

    def worst_individual_index(
        self, error_function: Callable[[Graph, Individual], float]
    ) -> int:
        """Returns the index of the individual with the largest error."""
        errors = self.individuals_errors(error_function)
        _return_value = max(range(len(errors)), key=errors.__getitem__)
        return _return_value

    def best_individual(
        self, error_function: Callable[[Graph, Individual], float]
    ) -> Individual:
        """Returns the individual with the smallest error."""
        _return_value = self._individuals[self.best_individual_index(error_function)]
        return _return_value

    def worst_individual(
        self, error_function: Callable[[Graph, Individual], float]
    ) -> Individual:
        """Returns the individual with the largest error."""
        _return_value = self._individuals[self.worst_individual_index(error_function)]
        return _return_value

    def individuals_errors(
        self, error_function: Callable[[Graph, Individual], float]
    ) -> List[float]:
        """Returns a list of individuals errors."""
        _return_value = [
            error_function(self._graph, individual) for individual in self.individuals
        ]
        return _return_value

    def min_error(self, error_function: Callable[[Graph, Individual], float]) -> float:
        """Returns the smallest error in the population."""
        _return_value = min(self.individuals_errors(error_function))
        return _return_value

    def max_error(self, error_function: Callable[[Graph, Individual], float]) -> float:
        """Returns the largest error in the population."""
        _return_value = max(self.individuals_errors(error_function))
        return _return_value

    def _calculate_metrics(self) -> bool:
        for individual in self.individuals:
            self._sum_conflicts_weight += individual.conflicts_weight
        return False

    def _update_metrics(self, ind: int, old_indv: Individual) -> bool:
        new_indv = self.individuals[ind]
        self._sum_conflicts_weight -= old_indv.conflicts_weight
        self._sum_conflicts_weight += new_indv.conflicts_weight

        best_conflicts_weight = self._best_individuals[ind].conflicts_weight
        new_conflicts_weight = new_indv.conflicts_weight
        if new_conflicts_weight < best_conflicts_weight:
            self._best_individuals[ind] = new_indv
        return False
