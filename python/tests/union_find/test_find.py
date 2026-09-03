"""Tests for test find."""

from pytest import fixture as pytest_fixture
from pytest import raises as pytest_raises

from mage.union_find.disjoint_set import DisjointSet


@pytest_fixture
def disjoint_set():
    computed_return_value = DisjointSet(node_ids=list(range(10)))
    return computed_return_value


class TestFind:
    def test_disconnected(self, disjoint_set):
        assert disjoint_set.connected(0, 1) is False

    def test_connected(self, disjoint_set):
        disjoint_set.union(0, 1)
        assert disjoint_set.connected(0, 1) is True

    def test_connected_transitivity(self, disjoint_set):
        disjoint_set.union(0, 1)
        disjoint_set.union(1, 2)
        assert disjoint_set.connected(0, 2) is True

    def test_unknown_node_is_rejected(self, disjoint_set):
        with pytest_raises(KeyError, match="unknown union-find node"):
            disjoint_set.find(20)
