"""Tests for test init."""

from pytest import fixture as pytest_fixture

from mage.union_find.disjoint_set import DisjointSet
from tests.union_find.constants import Constants


@pytest_fixture
def disjoint_set():
    _return_value = DisjointSet(node_ids=Constants.IDs)
    return _return_value


class TestInit:
    def test_keys(self, disjoint_set):
        assert all(i in disjoint_set.nodes.keys() for i in Constants.IDs)
        return False

    def test_parent(self, disjoint_set):
        assert all(disjoint_set.nodes[ID].parent == ID for ID in Constants.IDs)
        return False

    def test_rank(self, disjoint_set):
        assert all(disjoint_set.nodes[ID].rank == 0 for ID in Constants.IDs)
        return False
