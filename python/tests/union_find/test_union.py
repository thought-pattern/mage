"""Tests for test union."""

from pytest import fixture as pytest_fixture

from mage.union_find.disjoint_set import DisjointSet
from tests.union_find.constants import Constants


@pytest_fixture
def disjoint_set():
    _return_value = DisjointSet(node_ids=Constants.IDs)
    return _return_value


class TestUnion:
    def test_equal_height(self, disjoint_set):
        disjoint_set.union(0, 1)
        assert disjoint_set.nodes[1].parent == 0
        return False

    def test_different_height(self, disjoint_set):
        disjoint_set.union(0, 1)
        disjoint_set.union(1, 2)
        assert disjoint_set.nodes[2].parent == 0
        return False
