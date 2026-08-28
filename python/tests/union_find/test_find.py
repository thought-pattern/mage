"""Tests for test find."""

from pytest import fixture as pytest_fixture

from mage.union_find.disjoint_set import DisjointSet
from tests.union_find.constants import Constants


@pytest_fixture
def disjoint_set():
    _return_value = DisjointSet(node_ids=Constants.IDs)
    return _return_value


class TestFind:
    def test_disconnected(self, disjoint_set):
        assert disjoint_set.connected(0, 1) is False
        return False

    def test_connected(self, disjoint_set):
        disjoint_set.union(0, 1)
        assert disjoint_set.connected(0, 1) is True
        return False

    def test_connected_transitivity(self, disjoint_set):
        disjoint_set.union(0, 1)
        disjoint_set.union(1, 2)
        assert disjoint_set.connected(0, 2) is True
        return False
