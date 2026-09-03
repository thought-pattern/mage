"""Tests for test union."""

from pytest import fixture as pytest_fixture

from mage.union_find.disjoint_set import DisjointSet


@pytest_fixture
def disjoint_set():
    computed_return_value = DisjointSet(node_ids=list(range(10)))
    return computed_return_value


class TestUnion:
    def test_equal_height(self, disjoint_set):
        disjoint_set.union(0, 1)
        assert disjoint_set.connected(0, 1)
        assert disjoint_set.connected(0, 2) is False

    def test_different_height(self, disjoint_set):
        disjoint_set.union(0, 1)
        disjoint_set.union(2, 3)
        disjoint_set.union(1, 2)
        assert disjoint_set.connected(0, 3)
