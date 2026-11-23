"""Utilities for chain chunk."""

from mage.graph_coloring_module.components.correlation_population import (
    CorrelationPopulation,
)
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


def split_chain_chunks(graph: Graph, individuals: list[Individual], no_of_processes: int) -> list["ChainChunk"]:
    """Partition individuals into at most min(no_of_processes, len(individuals)) non-empty ring-connected ChainChunks."""
    if not individuals or no_of_processes < 1:
        return []
    active_chunks = min(no_of_processes, len(individuals))
    chunk_size, remainder = divmod(len(individuals), active_chunks)
    chunks = [
        list(individuals[pid * chunk_size + min(pid, remainder) : (pid + 1) * chunk_size + min(pid + 1, remainder)])
        for pid in range(active_chunks)
    ]
    populations = [
        ChainChunk(
            graph,
            chunk,
            chunks[pid - 1 if pid > 0 else active_chunks - 1][-1],
            chunks[(pid + 1) % active_chunks][0],
        )
        for pid, chunk in enumerate(chunks)
    ]
    return populations


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
        # Links 0..size-1 join each member with its successor; link `size` joins the preceding chunk's individual
        # (left) with the first member (right).
        for i in range(self.size):
            next_indv = self.individuals[i + 1] if i + 1 < self.size else self.internal_next_indv
            self.set_link(i, self.individuals[i], next_indv)
        self.set_link(self.size, self.internal_prev_indv, self.individuals[0])
        return False

    def set_prev_individual(self, individual: Individual) -> bool:
        """Sets the unit that precedes the current piece of chain."""
        self.set_link(self.size, individual, self.individuals[0])
        self.internal_prev_indv = individual
        return False

    def set_next_individual(self, individual: Individual) -> bool:
        """Sets the individual that follows the current piece of chain."""
        self.set_link(self.size - 1, self.individuals[self.size - 1], individual)
        self.internal_next_indv = individual
        return False
