"""Utilities for graph."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class Graph(ABC):
    def __init__(self, is_directed: bool):
        self._is_directed = is_directed
        self._nodes: List[int] = []
        self._preprocessed_transition_probs = {}
        self._first_pass_transition_probs = {}

    @property
    def is_directed(self):
        return self._is_directed

    @property
    def preprocessed_transition_probs(self) -> Dict[Tuple[int, int], List[float]]:
        return self._preprocessed_transition_probs

    @property
    def first_pass_transition_probs(self) -> Dict[int, List[float]]:
        return self._first_pass_transition_probs

    @property
    @abstractmethod
    def nodes(self) -> List[int]:
        return []

    @nodes.setter
    def nodes(self, value):
        self._nodes = value
        return False

    @preprocessed_transition_probs.setter
    def preprocessed_transition_probs(self, value):
        self._preprocessed_transition_probs = value
        return False

    @first_pass_transition_probs.setter
    def first_pass_transition_probs(self, value):
        self._first_pass_transition_probs = value
        return False

    @abstractmethod
    def has_edge(self, src_node_id: int, dest_node_id: int) -> bool: ...

    @abstractmethod
    def get_edge_weight(self, src_node_id: int, dest_node_id: int) -> float:
        return 0.0

    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[int]:
        return []

    @abstractmethod
    def get_edges(self) -> List[Tuple[int, int]]:
        return []

    @abstractmethod
    def set_edge_transition_probs(
        self, edge: Tuple[int, int], transition_probs: List[float]
    ) -> bool: ...

    @abstractmethod
    def get_edge_transition_probs(self, edge: Tuple[int, int]) -> List[float]:
        return []

    @abstractmethod
    def set_node_first_pass_transition_probs(
        self, source_node_id: int, normalized_probs: List[float]
    ) -> bool: ...

    @abstractmethod
    def get_node_first_pass_transition_probs(self, source_node_id: int) -> List[float]:
        return []


class GraphHolder(Graph):
    def __init__(self, edges_weights: Dict[Tuple[int, int], float], is_directed: bool):
        super().__init__(is_directed)
        self._edges_weights = edges_weights
        self._graph = {}
        self.init_graph()

    def nodes(self) -> List[int]:
        _return_value = list(self._graph.keys())
        return _return_value

    def set_edge_transition_probs(
        self, edge: Tuple[int, int], transition_probs: List[float]
    ) -> bool:
        self._preprocessed_transition_probs[edge] = transition_probs
        return False

    def get_edge_transition_probs(self, edge: Tuple[int, int]) -> List[float]:
        _return_value = self._preprocessed_transition_probs.get(edge, "")
        return _return_value

    def set_node_first_pass_transition_probs(
        self, source_node_id: int, normalized_probs: List[float]
    ) -> bool:
        self._first_pass_transition_probs[source_node_id] = normalized_probs
        return False

    def get_node_first_pass_transition_probs(self, source_node_id: int) -> List[float]:
        _return_value = self._first_pass_transition_probs.get(source_node_id, "")
        return _return_value

    def has_edge(self, src_node_id: int, dest_node_id: int) -> bool:
        _return_value = (src_node_id, dest_node_id) in self._edges_weights or (
            not self.is_directed and (dest_node_id, src_node_id) in self._edges_weights
        )
        return _return_value

    def get_edges(self) -> List[Tuple[int, int]]:
        edges = list(self._edges_weights.keys())
        if self._is_directed:
            return edges
        edges.extend([(edge[1], edge[0]) for edge in edges])
        return edges

    def get_edge_weight(self, src_node_id: int, dest_node_id: int) -> float:
        if not self.has_edge(src_node_id, dest_node_id):
            raise ValueError
        if (src_node_id, dest_node_id) in self._edges_weights:
            _return_value = self._edges_weights[(src_node_id, dest_node_id)]
            return _return_value
        _return_value = self._edges_weights[(dest_node_id, src_node_id)]
        return _return_value

    # Always return nodes in same order
    def get_neighbors(self, node_id: int) -> List[int]:
        _return_value = (
            self._graph.get(node_id, False) if node_id in self._graph else []
        )
        return _return_value

    def init_graph(self) -> bool:
        for node_from, node_to in self._edges_weights:
            if node_from not in self._graph:
                self._graph[node_from] = set()
            self._graph.get(node_from, set()).add(node_to)
            if not self.is_directed:
                if node_to not in self._graph:
                    self._graph[node_to] = set()
                self._graph.get(node_to, set()).add(node_from)

        self.nodes = list(self._graph.keys())

        for node in self._graph:
            self._graph[node] = sorted(list(self._graph.get(node, [])))
        return False
