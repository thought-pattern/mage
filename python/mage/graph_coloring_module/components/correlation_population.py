"""Utilities for correlation population."""

from abc import abstractmethod
from collections import Counter as collections_Counter

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph


def same_key_pairs(counts: dict) -> int:
    """Returns the number of unordered node pairs that share a key, given how many nodes hold each key."""
    pairs = sum(count * (count - 1) // 2 for count in counts.values())
    return pairs


def pair_terms_delta(old_peers: int, old_partner_peers: int, new_peers: int, new_partner_peers: int) -> int:
    """Correlation change on one link when one node moves from its old color to a new color. Its pair term with each
    peer that held the old color goes from same (-1) to different (+1), and with each peer holding the new color the
    other way; each change is weighted by the partner's sign for that pair, which is -1 for the peers that also share
    the node's color in the partner (the *_partner_peers counts)."""
    delta = 2 * (old_peers - 2 * old_partner_peers) - 2 * (new_peers - 2 * new_partner_peers)
    return delta


class CorrelationPopulation(Population):
    """A population whose neighboring individuals are joined by correlation links. The correlation of a link is
    the sum, over every unordered pair of distinct nodes, of S_left * S_right, where S is -1 when the individual gives
    both nodes one color and 1 otherwise.

    Link orientation contract: each link pairs a left and a right individual in chain order, and the individual at
    index i is the left end of link get_next_correlation_index(i) and the right end of link
    get_prev_correlation_index(i). Each link keeps the number of nodes per (left color, right color) and each member
    keeps the number of nodes per color, so a recoloring updates the correlation exactly from counts, and full and
    incremental values never drift apart."""

    def __init__(self, graph: Graph, individuals: list[Individual]):
        super().__init__(graph, individuals)
        self.internal_cumulative_correlation = 0
        self.internal_correlation = []
        self.internal_link_color_pairs: list[dict[tuple[int, int], int]] = []
        self.internal_color_counts = [collections_Counter(individual.chromosome) for individual in individuals]

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

    def set_link(self, link_index: int, left: Individual, right: Individual) -> bool:
        """Computes link `link_index` between left and right from scratch and installs it (appending the next link
        or replacing an existing one). Expanding S = 1 - 2 * same gives the all-pairs sum as
        pairs - 2 * same_left - 2 * same_right + 4 * same_both, evaluated from color and paired-color counts."""
        node_count = len(left.chromosome)
        color_pairs = collections_Counter(zip(left.chromosome, right.chromosome, strict=True))
        correlation = (
            node_count * (node_count - 1) // 2
            - 2 * same_key_pairs(collections_Counter(left.chromosome))
            - 2 * same_key_pairs(collections_Counter(right.chromosome))
            + 4 * same_key_pairs(color_pairs)
        )

        if link_index == len(self.internal_correlation):
            self.internal_correlation.append(correlation)
            self.internal_link_color_pairs.append(color_pairs)
        else:
            self.internal_cumulative_correlation -= self.internal_correlation[link_index]
            self.internal_correlation[link_index] = correlation
            self.internal_link_color_pairs[link_index] = color_pairs
        self.internal_cumulative_correlation += correlation
        return False

    def update_correlation(self, index: int, old_individual: Individual, nodes: list[int]) -> int:
        next_correlation_index = self.get_next_correlation_index(index)
        prev_correlation_index = self.get_prev_correlation_index(index)

        new_individual = self.individuals[index]
        prev_individual = self.get_prev_individual(index)
        next_individual = self.get_next_individual(index)

        if prev_individual is new_individual or next_individual is new_individual:
            # A one-member ring links the individual with itself, so both ends of its link moved; recompute it.
            cumulative_before = self.internal_cumulative_correlation
            self.internal_color_counts[index] = collections_Counter(new_individual.chromosome)
            self.set_link(prev_correlation_index, prev_individual, new_individual)
            self.set_link(next_correlation_index, new_individual, next_individual)
            delta_corr = self.internal_cumulative_correlation - cumulative_before
            return delta_corr

        color_counts = self.internal_color_counts[index]
        prev_pairs = self.internal_link_color_pairs[prev_correlation_index]
        next_pairs = self.internal_link_color_pairs[next_correlation_index]

        correlation_prev_delta = 0
        correlation_next_delta = 0
        # Each changed node is applied once, in sequence, so every step is an exact single-node move from the
        # current counts; repeated IDs from a mutation cannot count a pair twice.
        for node in set(nodes):
            old_color = old_individual[node]
            new_color = new_individual[node]
            if old_color == new_color:
                continue
            prev_color = prev_individual[node]
            next_color = next_individual[node]

            old_peers = color_counts.get(old_color, 0) - 1
            new_peers = color_counts.get(new_color, 0)
            prev_old_peers = prev_pairs.get((prev_color, old_color), 0) - 1
            prev_new_peers = prev_pairs.get((prev_color, new_color), 0)
            next_old_peers = next_pairs.get((old_color, next_color), 0) - 1
            next_new_peers = next_pairs.get((new_color, next_color), 0)

            correlation_prev_delta += pair_terms_delta(old_peers, prev_old_peers, new_peers, prev_new_peers)
            correlation_next_delta += pair_terms_delta(old_peers, next_old_peers, new_peers, next_new_peers)

            color_counts[old_color] = old_peers
            color_counts[new_color] = new_peers + 1
            prev_pairs[(prev_color, old_color)] = prev_old_peers
            prev_pairs[(prev_color, new_color)] = prev_new_peers + 1
            next_pairs[(old_color, next_color)] = next_old_peers
            next_pairs[(new_color, next_color)] = next_new_peers + 1

        self.internal_correlation[prev_correlation_index] += correlation_prev_delta
        self.internal_correlation[next_correlation_index] += correlation_next_delta
        delta_corr = correlation_prev_delta + correlation_next_delta
        self.internal_cumulative_correlation += delta_corr
        return delta_corr
