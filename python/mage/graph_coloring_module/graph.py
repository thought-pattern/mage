"""Utilities for graph."""



class Graph:
    """A data structure representing an undirected weighted graph.

    :param nodes: a list containing the labels of all nodes in the graph
    :param adjacency_list: a dictionary that associates each node with a list of its neighbors
    :name: a name of the graph

    """

    def __init__(
        self,
        nodes: list,
        adjacency_list: dict,
        name: str = "",
    ):
        if len(set(nodes)) != len(nodes):
            raise ValueError("graph node labels must be unique and hashable")
        self.indices_to_labels = list(nodes)
        self.labels_to_indices = dict((label, index) for index, label in enumerate(nodes))
        self.nodes_count = len(nodes)
        self.neighbors_positions: list[int] = []
        self.internal_neighbors: list[int] = []
        self.weights: list[float] = []
        self.internal_name = name

        for label in self.indices_to_labels:
            neighbors = adjacency_list.get(label, [])
            if not isinstance(neighbors, list):
                raise TypeError(f"adjacency entries for {label!r} must be lists")
            for neighbor in neighbors:
                if not isinstance(neighbor, tuple) or len(neighbor) != 2:
                    raise TypeError(f"adjacency entries for {label!r} must be (label, weight) pairs")
                neighbor_index = self.labels_to_indices.get(neighbor[0], False)
                if not isinstance(neighbor_index, int) or isinstance(neighbor_index, bool):
                    raise ValueError(f"adjacency entry for {label!r} references unknown node {neighbor[0]!r}")
                weight = neighbor[1]
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise TypeError(f"edge weight for {label!r} -> {neighbor[0]!r} must be numeric")
                self.internal_neighbors.append(neighbor_index)
                self.weights.append(float(weight))
            self.neighbors_positions.append(len(self.internal_neighbors))

    def __str__(self):
        """Returns the name of the graph. If the name is not given
        at initialization, returns an empty string."""
        return self.internal_name

    def __len__(self):
        """Returns the number of nodes in the graph."""
        return self.nodes_count

    def __getitem__(self, node: int) -> list[int]:
        """Returns an iterator over neighbors of the given node."""
        if not isinstance(node, int) or isinstance(node, bool) or not 0 <= node < self.nodes_count:
            raise IndexError(f"graph node index out of range: {node}")
        start = self.neighbors_positions[node - 1] if node != 0 else 0
        end = self.neighbors_positions[node]
        computed_return_value = self.internal_neighbors[start:end]
        return computed_return_value

    @property
    def nodes(self) -> range:
        """Returns an iterator over nodes in the graph."""
        nodes = range(self.nodes_count)
        return nodes

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self.nodes_count

    def number_of_edges(self) -> int:
        """Returns the number of edges in the graph."""
        computed_return_value = len(self.internal_neighbors) // 2
        return computed_return_value

    def neighbors(self, node: int) -> list[int]:
        """Returns an iterator over neighbors of node n."""
        computed_return_value = self.__getitem__(node)
        return computed_return_value

    def weighted_neighbors(self, node: int) -> list[tuple[int, float]]:
        """Returns an iterator over neighbor and weight tuples of the node."""
        start = self.neighbors_positions[node - 1] if node != 0 else 0
        end = self.neighbors_positions[node]
        computed_return_value = self.neighbor_weight_tuples(start, end)
        return computed_return_value

    def weight(self, node_1: int, node_2: int) -> float:
        """Returns the weight between the two given nodes."""
        weighted_neighs = self.weighted_neighbors(node_1)
        for node, weight in weighted_neighs:
            if node == node_2:
                return weight
        return 0.0

    def degree(self, node: int) -> int:
        """Returns the degree of the given node."""
        start = self.neighbors_positions[node - 1] if node != 0 else 0
        end = self.neighbors_positions[node]
        computed_return_value = end - start
        return computed_return_value

    def label(self, node: int) -> object:
        """Returns the node label."""
        computed_return_value = self.indices_to_labels[node]
        return computed_return_value

    def neighbor_weight_tuples(self, start: int, end: int) -> list[tuple[int, float]]:
        computed_return_value = [(self.internal_neighbors[index], self.weights[index]) for index in range(start, end)]
        return computed_return_value
