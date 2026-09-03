"""Utilities for individual."""

from random import choice

from mage.graph_coloring_module.exceptions import (
    IllegalColorException,
    IllegalNodeException,
    WrongColoringException,
)
from mage.graph_coloring_module.graph import Graph


class Individual:
    """A class that represents an individual. The individual represents
    one possible coloring of the graph. Individual also contains data about
    conflicts, like the sum of weights of conflict edges, and set of conflict nodes.
    If a new individual is created by changing the color of some nodes of
    the current individual then this data is calculated based on the data
    of the current individual."""

    def __init__(
        self,
        no_of_colors: int,
        graph: Graph,
        chromosome: object = False,
        conflicts_weight: object = 0,
        conflict_nodes: object = False,
        conflicts_counter: object = False,
    ):
        if not isinstance(no_of_colors, int) or isinstance(no_of_colors, bool) or no_of_colors <= 0:
            raise ValueError("no_of_colors must be a positive integer")
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph")
        if not isinstance(conflicts_weight, (int, float)) or isinstance(conflicts_weight, bool):
            raise TypeError("conflicts_weight must be numeric")
        self.internal_graph = graph
        self.internal_no_of_units = len(graph)
        self.internal_no_of_colors = no_of_colors
        if chromosome is False:
            self.internal_chromosome = list(choice(range(no_of_colors)) for _ in range(len(graph)))
        else:
            if not isinstance(chromosome, list) or len(chromosome) != len(graph):
                raise ValueError("chromosome must contain one color per graph node")
            if not all(
                isinstance(color, int) and not isinstance(color, bool) and 0 <= color < no_of_colors for color in chromosome
            ):
                raise ValueError("chromosome colors must be integers in the allowed color range")
            self.internal_chromosome = list(chromosome)

        self.internal_conflicts_weight = float(conflicts_weight)
        valid_counter = (
            isinstance(conflicts_counter, list)
            and len(conflicts_counter) == len(graph)
            and all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in conflicts_counter)
        )
        valid_nodes = isinstance(conflict_nodes, set) and all(
            isinstance(node, int) and not isinstance(node, bool) and 0 <= node < len(graph) for node in conflict_nodes
        )
        if valid_counter and valid_nodes and isinstance(conflicts_counter, list) and isinstance(conflict_nodes, set):
            self.internal_conflicts_counter = list(conflicts_counter)
            self.internal_conflict_nodes = set(conflict_nodes)
        else:
            self.internal_conflicts_counter = []
            self.internal_conflict_nodes = set()

        if not valid_counter or not valid_nodes:
            self.calculate_conflicts()

    def __getitem__(self, index: int) -> int:
        """Returns the color stored on the given index."""
        computed_return_value = self.internal_chromosome[index]
        return computed_return_value

    @property
    def chromosome(self) -> list[int]:
        """Returns the list representing the coloring of the graph."""
        return self.internal_chromosome

    @property
    def conflict_nodes(self) -> set[int]:
        """Returns a set of conflicting nodes in the coloring
        represented by the individual.."""
        return self.internal_conflict_nodes

    @property
    def graph(self) -> Graph:
        """Returns the graph whose coloring the individual represents."""
        return self.internal_graph

    @property
    def no_of_colors(self) -> int:
        """Returns the allowed number of colors."""
        return self.internal_no_of_colors

    @property
    def no_of_units(self) -> int:
        """Returns the size of the chromosome."""
        return self.internal_no_of_units

    @property
    def conflicts_weight(self) -> float:
        """Returns the sum of weights of conflicting edges
        in the coloring represented by the individual."""
        return self.internal_conflicts_weight

    def check_coloring(self) -> bool:
        """Checks that the coloring represented by the individual is correct.
        The coloring is correct if it does not color two nodes connected with
        an edge with the same color. The function returns True if the coloring
        is correct, otherwise returns False."""
        for node in self.graph.nodes:
            for neigh in self.graph[node]:
                if self.chromosome[node] == self.chromosome[neigh]:
                    return False
        return True

    def replace_unit(self, index: int, color: int):
        """Sets the color of the node on the given index to the given color and
        returns a new individual if the given arguments are correct. If the given
        color is not allowed then the IllegalColorException exception is raised.
        If the given node does not exist then the IllegalNodeException is raised."""
        computed_return_value = self.replace_units([index], [color])
        return computed_return_value

    def replace_units(self, indices: list[int], colors: list[int]):
        """Sets the colors of the nodes with the corresponding indices to the given
        colors and returns a new individual if the given coloring is correct. If any
        of the given colors is not allowed then the IllegalColorException exception is
        raised. If any of the given nodes does not exist then the IllegalNodeException
        is raised. If the number of given nodes is not equal to the number of given colors
        then the WrongColoringException is raised."""

        if len(indices) != len(colors):
            raise WrongColoringException("The number of given nodes must be equal to the number of given colors!")

        new_chromosome = self.internal_chromosome[:]
        conflicts_counter = self.internal_conflicts_counter[:]
        conflict_nodes = self.internal_conflict_nodes.copy()
        conflict_edges = self.conflicts_weight

        for index, color in zip(indices, colors, strict=False):
            if not (0 <= color < self.internal_no_of_colors):
                raise IllegalColorException("The given color is not in the range of allowed colors!")
            if not (0 <= index < self.no_of_units):
                raise IllegalNodeException("The given node does not exist!")
            conflict_edges, conflicts_counter, conflict_nodes = self.calculate_diff(
                chromosome=new_chromosome,
                node=index,
                color=color,
                conflict_edges=conflict_edges,
                conflicts_counter=conflicts_counter,
                conflict_nodes=conflict_nodes,
            )
            new_chromosome[index] = color

        new_indv = Individual(
            no_of_colors=self.no_of_colors,
            graph=self.graph,
            chromosome=new_chromosome,
            conflicts_weight=conflict_edges,
            conflict_nodes=conflict_nodes,
            conflicts_counter=conflicts_counter,
        )

        return new_indv

    def calculate_diff(
        self,
        chromosome: list[int],
        node: int,
        color: int,
        conflict_edges: float,
        conflicts_counter: list[int],
        conflict_nodes: set[int],
    ) -> tuple[float, list[int], set[int]]:
        diff = 0
        for neigh, weight in self.graph.weighted_neighbors(node):
            if chromosome[node] == chromosome[neigh]:
                if not (color == chromosome[neigh]):
                    diff -= weight

                    conflicts_counter[neigh] -= 1
                    if conflicts_counter[neigh] == 0:
                        conflict_nodes.remove(neigh)

                    conflicts_counter[node] -= 1
                    if conflicts_counter[node] == 0:
                        conflict_nodes.remove(node)
            else:
                if color == chromosome[neigh]:
                    diff += weight

                    conflicts_counter[neigh] += 1
                    if conflicts_counter[neigh] == 1:
                        conflict_nodes.add(neigh)

                    conflicts_counter[node] += 1
                    if conflicts_counter[node] == 1:
                        conflict_nodes.add(node)

        conflict_edges = conflict_edges + diff
        return conflict_edges, conflicts_counter, conflict_nodes

    def calculate_conflicts(self):
        self.internal_conflict_nodes = set()
        self.internal_conflicts_counter = [0 for _ in self.graph.nodes]
        conflicting_edges = 0.0

        for node in self.graph.nodes:
            for neigh, weight in self.graph.weighted_neighbors(node):
                if self.chromosome[node] == self.chromosome[neigh]:
                    self.internal_conflicts_counter[node] += 1
                    conflicting_edges += weight
                    if self.internal_conflicts_counter[node] == 1:
                        self.internal_conflict_nodes.add(node)

        conflicting_edges //= 2
        self.internal_conflicts_weight = conflicting_edges
        return False
