"""Tests for test init."""

from pytest import fixture as pytest_fixture
from pytest import raises as pytest_raises

from mage.union_find.disjoint_set import DisjointSet


@pytest_fixture
def disjoint_set():
    computed_return_value = DisjointSet(node_ids=list(range(10)))
    return computed_return_value


class TestInit:
    def test_nodes_begin_as_independent_components(self, disjoint_set):
        assert all(disjoint_set.connected(node_id, node_id) for node_id in range(10))
        assert disjoint_set.connected(0, 9) is False

    def test_reinitialize_replaces_components(self, disjoint_set):
        disjoint_set.union(0, 1)
        disjoint_set.reinitialize([4, 5])

        assert disjoint_set.connected(4, 5) is False
        with pytest_raises(KeyError):
            disjoint_set.find(0)

    def test_duplicate_nodes_are_rejected(self):
        with pytest_raises(ValueError, match="unique"):
            DisjointSet([1, 1])
