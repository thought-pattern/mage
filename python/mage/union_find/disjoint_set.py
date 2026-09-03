"""Utilities for disjoint set."""

from mage.union_find.node import INITIAL_RANK, Node


class DisjointSet:
    """
    Class implementing a disjoint-set data structure.
    """

    def __init__(self, node_ids: object = False):
        if node_ids is False:
            self.nodes: dict[int, Node] = {}
            return
        self.validate_node_ids(node_ids)
        if not isinstance(node_ids, list):
            raise TypeError("node_ids must be a list of integers")
        self.nodes = {node_id: Node(parent_id=node_id, rank=INITIAL_RANK) for node_id in node_ids}

    @staticmethod
    def validate_node_ids(node_ids: object) -> bool:
        if not isinstance(node_ids, list):
            raise TypeError("node_ids must be a list of integers")
        if not all(isinstance(node_id, int) and not isinstance(node_id, bool) for node_id in node_ids):
            raise TypeError("node_ids must contain only integers")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique")
        return True

    def reinitialize(self, node_ids: list[int]):
        """
        Reinitializes the data structure.

        :param node_ids: NodeIDs to be included in the structure.
        :type node_ids: List[int]
        """
        self.validate_node_ids(node_ids)
        self.nodes = {node_id: Node(parent_id=node_id, rank=INITIAL_RANK) for node_id in node_ids}
        return False

    def parent(self, node_id: int) -> int:
        """
        Returns given node's parent's ID.

        :param node_id: Node ID
        :type node_id: int
        """
        node = self.nodes.get(node_id, False)
        if not isinstance(node, Node):
            raise KeyError(f"unknown union-find node {node_id}")
        return node.parent

    def grandparent(self, node_id: int) -> int:
        """
        Returns given node's grandparent's ID.

        :param node_id: Node ID
        :type node_id: int
        """
        computed_return_value = self.parent(self.parent(node_id))
        return computed_return_value

    def rank(self, node_id: int) -> int:
        """
        Returns given node's rank.

        :param node_id: Node ID
        :type node_id: int
        """
        node = self.nodes.get(node_id, False)
        if not isinstance(node, Node):
            raise KeyError(f"unknown union-find node {node_id}")
        return node.rank

    def find(self, node_id: int) -> int:
        """
        Returns the representative node's ID for the component that given node is member of.
        Uses path splitting (https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives)
        in order to keep trees representing connected components flat.

        :param node_id: Node ID
        :type node_id: int
        """
        while node_id != self.parent(node_id):
            node = self.nodes.get(node_id, False)
            if not isinstance(node, Node):
                raise KeyError(f"unknown union-find node {node_id}")
            node.parent = self.grandparent(node_id)
            node_id = self.parent(node_id)

        return node_id

    def union(self, node1_id: int, node2_id: int) -> bool:
        """
        Unites the components containing two given nodes. Implements union by rank to reduce component tree height.

        :param node1_id: First node's ID
        :type node1_id: int
        :param node2_id: Second node's ID
        :type node2_id: int
        """
        root_1 = self.find(node1_id)
        root_2 = self.find(node2_id)

        if root_1 == root_2:
            return False

        if self.rank(root_1) < self.rank(root_2):
            root_1, root_2 = root_2, root_1

        root_2_node = self.nodes.get(root_2, False)
        root_1_node = self.nodes.get(root_1, False)
        if not isinstance(root_2_node, Node) or not isinstance(root_1_node, Node):
            raise KeyError("union-find root disappeared during union")
        root_2_node.parent = root_1
        if self.rank(root_1) == self.rank(root_2):
            root_1_node.rank = self.rank(root_1) + 1
        return False

    def connected(self, node1_id: int, node2_id: int) -> bool:
        """
        Returns whether two given nodes belong to the same connected component.

        :param node1_id: First node's ID
        :type node1_id: int
        :param node2_id: Second node's ID
        :type node2_id: int
        """
        computed_return_value = self.find(node1_id) == self.find(node2_id)
        return computed_return_value
