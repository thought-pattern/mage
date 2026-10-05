"""Utilities for graph."""

from abc import ABC, abstractmethod


class Graph(ABC):
    def __init__(self, is_directed: bool):
        self.internal_is_directed = is_directed
        self.internal_nodes: list[int] = []
        self.internal_preprocessed_transition_probs = {}
        self.internal_first_pass_transition_probs = {}

    @property
    def is_directed(self):
        return self.internal_is_directed

    @property
    def preprocessed_transition_probs(self) -> dict[tuple[int, int], list[float]]:
        return self.internal_preprocessed_transition_probs

    @property
    def first_pass_transition_probs(self) -> dict[int, list[float]]:
        return self.internal_first_pass_transition_probs

    @property
    @abstractmethod
    def nodes(self) -> list[int]:
        return []

    @nodes.setter
    def nodes(self, value):
        self.internal_nodes = value
        return False

    @preprocessed_transition_probs.setter
    def preprocessed_transition_probs(self, value):
        self.internal_preprocessed_transition_probs = value
        return False

    @first_pass_transition_probs.setter
    def first_pass_transition_probs(self, value):
        self.internal_first_pass_transition_probs = value
        return False

    @abstractmethod
    def has_edge(self, src_node_id: int, dest_node_id: int) -> bool: ...

    @abstractmethod
    def get_edge_weight(self, src_node_id: int, dest_node_id: int) -> float:
        return 0.0

    @abstractmethod
    def get_neighbors(self, node_id: int) -> list[int]:
        return []

    @abstractmethod
    def get_edges(self) -> list[tuple[int, int]]:
        return []

    @abstractmethod
    def set_edge_transition_probs(self, edge: tuple[int, int], transition_probs: list[float]) -> bool: ...

    @abstractmethod
    def get_edge_transition_probs(self, edge: tuple[int, int]) -> list[float]:
        return []

    @abstractmethod
    def set_node_first_pass_transition_probs(self, source_node_id: int, normalized_probs: list[float]) -> bool: ...

    @abstractmethod
    def get_node_first_pass_transition_probs(self, source_node_id: int) -> list[float]:
        return []


class GraphHolder(Graph):
    def __init__(self, edges_weights: dict[tuple[int, int], float], is_directed: bool, declared_nodes=()):
        """
        Args:
            edges_weights: weight per (from, to) edge.
            is_directed: whether edges are directed.
            declared_nodes: every vertex ID of the source graph. Edges cannot reveal isolated vertices, so the
                graph-conversion boundary declares them; endpoints of edges are nodes whether declared or not.
        """
        super().__init__(is_directed)
        self.internal_edges_weights = edges_weights
        self.internal_declared_nodes = list(declared_nodes)
        self.internal_graph = {}
        self.init_graph()

    @property
    def nodes(self) -> list[int]:
        computed_return_value = list(self.internal_graph.keys())
        return computed_return_value

    def set_edge_transition_probs(self, edge: tuple[int, int], transition_probs: list[float]) -> bool:
        self.internal_preprocessed_transition_probs[edge] = transition_probs
        return False

    def get_edge_transition_probs(self, edge: tuple[int, int]) -> list[float]:
        computed_return_value = self.internal_preprocessed_transition_probs.get(edge, [])
        return computed_return_value

    def set_node_first_pass_transition_probs(self, source_node_id: int, normalized_probs: list[float]) -> bool:
        self.internal_first_pass_transition_probs[source_node_id] = normalized_probs
        return False

    def get_node_first_pass_transition_probs(self, source_node_id: int) -> list[float]:
        computed_return_value = self.internal_first_pass_transition_probs.get(source_node_id, [])
        return computed_return_value

    def has_edge(self, src_node_id: int, dest_node_id: int) -> bool:
        computed_return_value = (src_node_id, dest_node_id) in self.internal_edges_weights or (
            not self.is_directed and (dest_node_id, src_node_id) in self.internal_edges_weights
        )
        return computed_return_value

    def get_edges(self) -> list[tuple[int, int]]:
        """Returns every oriented edge exactly once; an undirected edge stored in one orientation is also returned
        reversed, and one stored in both orientations is not repeated."""
        edges = list(self.internal_edges_weights.keys())
        if self.internal_is_directed:
            return edges
        edges.extend([(edge[1], edge[0]) for edge in edges if (edge[1], edge[0]) not in self.internal_edges_weights])
        return edges

    def get_edge_weight(self, src_node_id: int, dest_node_id: int) -> float:
        if not self.has_edge(src_node_id, dest_node_id):
            raise ValueError
        # has_edge established that the stored orientation or, for an undirected graph, the reverse one is present.
        edge = (src_node_id, dest_node_id)
        if edge not in self.internal_edges_weights:
            edge = (dest_node_id, src_node_id)
        computed_return_value = self.internal_edges_weights.get(edge, 0.0)
        return computed_return_value

    # Always return nodes in same order
    def get_neighbors(self, node_id: int) -> list[int]:
        computed_return_value = self.internal_graph.get(node_id, [])
        return computed_return_value

    def init_graph(self) -> bool:
        # Declared vertices and both endpoints of every edge are nodes. A directed sink or an isolated vertex keeps an
        # empty neighbor list, so walks still start from it.
        for node in self.internal_declared_nodes:
            if node not in self.internal_graph:
                self.internal_graph[node] = set()
        for node_from, node_to in self.internal_edges_weights:
            if node_from not in self.internal_graph:
                self.internal_graph[node_from] = set()
            if node_to not in self.internal_graph:
                self.internal_graph[node_to] = set()
            self.internal_graph.get(node_from, set()).add(node_to)
            if not self.is_directed:
                self.internal_graph.get(node_to, set()).add(node_from)

        self.internal_nodes = list(self.internal_graph.keys())

        for node in self.internal_graph:
            self.internal_graph[node] = sorted(list(self.internal_graph.get(node, [])))
        return False
