"""Utilities for temporal neighborhood."""

from typing import Dict, List, Tuple

from numpy import append as np_append
from numpy import array as np_array
from numpy import random as np_random
from numpy import where as np_where
from numpy import zeros as np_zeros


class TemporalNeighborhood:
    def __init__(self):
        super().__init__()
        self.init_temporal_neighborhood()

    def init_temporal_neighborhood(self):
        self.neighborhood: Dict[int, List[Tuple[int, int, float]]] = {}
        return False

    def update_neighborhood(
        self,
        sources: np_array,
        destinations: np_array,
        edge_idxs: np_array,
        timestamps: np_array,
    ) -> bool:
        """
        Idea is that smallest new timestamp is always greater than last biggest one in dict so we don't need to
        sort arrays :)
        if it doesn't exist, create empty, else overwrite
        """
        self.neighborhood = {
            **{node: [] for node in set(sources).union(set(destinations))},
            **self.neighborhood,
        }
        for source, destination, edge_idx, timestamp in zip(
            sources, destinations, edge_idxs, timestamps, strict=False
        ):
            self.neighborhood.get(source, []).append((destination, edge_idx, timestamp))
            self.neighborhood.get(destination, []).append((source, edge_idx, timestamp))
        return False

    def get_neighborhood(
        self, node: int, timestamp: int, num_neighbors: int
    ) -> Tuple[np_array, np_array, np_array]:
        """ """
        if node not in self.neighborhood:
            _return_value = (
                np_zeros(num_neighbors, dtype=int),
                np_zeros(num_neighbors, dtype=int),
                np_zeros(num_neighbors, dtype=int),
            )
            return _return_value
        neighbors_tuple = self.neighborhood.get(node, False)

        neighbors, edge_idxs, timestamps = list(zip(*neighbors_tuple, strict=False))
        neighbors, edge_idxs, timestamps = (
            list(neighbors),
            list(edge_idxs),
            list(timestamps),
        )

        neighbors = np_array(neighbors)
        edge_idxs = np_array(edge_idxs)
        timestamps = np_array(timestamps)

        indices = np_where(timestamps < timestamp)[0]
        indices = np_random.choice(
            indices, size=min(num_neighbors, len(indices)), replace=False
        )

        neighbors = neighbors[indices]
        edge_idxs = edge_idxs[indices]
        timestamps = timestamps[indices]

        neighbors = np_append(
            arr=neighbors, values=np_zeros(num_neighbors - len(neighbors), dtype=int)
        )
        edge_idxs = np_append(
            arr=edge_idxs, values=np_zeros(num_neighbors - len(edge_idxs), dtype=int)
        )
        timestamps = np_append(
            arr=timestamps, values=np_zeros(num_neighbors - len(timestamps), dtype=int)
        )
        return neighbors, edge_idxs, timestamps

    def find_neighborhood(
        self, nodes: List[int], num_neighbors: int
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        _return_value = {
            node: self.neighborhood.get(node, [])[:num_neighbors] for node in nodes
        }
        return _return_value
