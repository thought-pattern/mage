"""Utilities for temporal neighborhood."""

from bisect import bisect_left, insort
from operator import itemgetter
from random import sample as random_sample

from numpy import array as np_array, ndarray as np_ndarray

NEIGHBOR_TIMESTAMP = itemgetter(2)


class TemporalNeighborhood:
    def __init__(self):
        super().__init__()
        self.init_temporal_neighborhood()

    def init_temporal_neighborhood(self):
        # node -> (neighbor, edge_idx, timestamp) entries kept in chronological order, so the neighbors eligible
        # before any time form a prefix that can be located by binary search
        self.neighborhood: dict[int, list[tuple[int, int, float]]] = {}
        return False

    def update_neighborhood(
        self,
        sources: np_ndarray,
        destinations: np_ndarray,
        edge_idxs: np_ndarray,
        timestamps: np_ndarray,
    ) -> bool:
        if not len(sources) == len(destinations) == len(edge_idxs) == len(timestamps):
            raise ValueError("Sources, destinations, edge indexes and timestamps must have the same length")
        for source, destination, edge_idx, timestamp in zip(sources, destinations, edge_idxs, timestamps, strict=True):
            self.insert_neighbor(source, (destination, edge_idx, timestamp))
            self.insert_neighbor(destination, (source, edge_idx, timestamp))
        return False

    def insert_neighbor(self, node: int, entry: tuple[int, int, float]) -> bool:
        neighbors = self.neighborhood.get(node, [])
        # A chronological stream appends; an earlier event is placed after neighbors with an equal or earlier time.
        if neighbors and NEIGHBOR_TIMESTAMP(neighbors[-1]) > NEIGHBOR_TIMESTAMP(entry):
            insort(neighbors, entry, key=NEIGHBOR_TIMESTAMP)
        else:
            neighbors.append(entry)
        self.neighborhood[node] = neighbors
        return False

    def get_neighborhood(self, node: int, timestamp: int, num_neighbors: int) -> tuple[np_ndarray, np_ndarray, np_ndarray]:
        """
        Samples, uniformly without replacement, at most num_neighbors of the node's interactions strictly before
        timestamp. Only real interactions are returned: a node with fewer eligible interactions yields shorter arrays,
        never placeholder node/edge identities. Work is O(log d + num_neighbors) in the node's history length d.

        :return: neighbors, edge indexes and timestamps of the sampled interactions, in chronological order
        """
        if num_neighbors < 1:
            raise ValueError(f"num_neighbors must be positive, received {num_neighbors}")
        neighbors = self.neighborhood.get(node, [])
        eligible_count = bisect_left(neighbors, timestamp, key=NEIGHBOR_TIMESTAMP)
        positions = sorted(random_sample(range(eligible_count), min(num_neighbors, eligible_count)))
        sampled = [neighbors[position] for position in positions]

        sampled_neighbors = np_array([entry[0] for entry in sampled], dtype=int)
        sampled_edge_idxs = np_array([entry[1] for entry in sampled], dtype=int)
        sampled_timestamps = np_array([entry[2] for entry in sampled], dtype=int)
        sampled_neighborhood = (sampled_neighbors, sampled_edge_idxs, sampled_timestamps)
        return sampled_neighborhood

    def find_neighborhood(self, nodes: list[int], num_neighbors: int) -> dict[int, list[tuple[int, int, float]]]:
        computed_return_value = {node: self.neighborhood.get(node, [])[:num_neighbors] for node in nodes}
        return computed_return_value
