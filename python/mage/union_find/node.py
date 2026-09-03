"""Utilities for node."""

INITIAL_RANK = 0


class Node:
    """
    Class implementing a node in an union-find data structure.
    Stores the current node's rank and a reference to its parent node.
    """

    def __init__(self, parent_id: int, rank: int = INITIAL_RANK):
        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            raise TypeError("parent_id must be an integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self.internal_parent = parent_id
        self.internal_rank = rank

    @property
    def parent(self) -> int:
        return self.internal_parent

    @parent.setter
    def parent(self, x: int):
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError("parent must be an integer")
        self.internal_parent = x
        return False

    @property
    def rank(self) -> int:
        return self.internal_rank

    @rank.setter
    def rank(self, x: int):
        if not isinstance(x, int) or isinstance(x, bool) or x < 0:
            raise ValueError("rank must be a non-negative integer")
        self.internal_rank = x
        return False
