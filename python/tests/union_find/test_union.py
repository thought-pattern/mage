"""Tests for test union."""

from mage.union_find.disjoint_set import DisjointSet


class TestUnion:
    def test_equal_height(self):
        disjoint_set = DisjointSet(node_ids=list(range(10)))
        disjoint_set.union(0, 1)
        assert disjoint_set.connected(0, 1)
        assert disjoint_set.connected(0, 2) is False

    def test_different_height(self):
        disjoint_set = DisjointSet(node_ids=list(range(10)))
        disjoint_set.union(0, 1)
        disjoint_set.union(2, 3)
        disjoint_set.union(1, 2)
        assert disjoint_set.connected(0, 3)
