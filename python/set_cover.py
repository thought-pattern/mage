"""Utilities for set cover."""

from abc import ABC as abc_ABC
from abc import abstractmethod as abc_abstractmethod

from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc

from mage.constraint_programming import (
    GekkoMatchingProblem,
    GekkoMPSolver,
    GreedyMatchingProblem,
    GreedyMPSolver,
)


@mgp_read_proc
def cp_solve(
    context: mgp_ProcCtx,
    element_vertexes: list[mgp_Vertex],
    set_vertexes: list[mgp_Vertex],
) -> list[mgp_Record]:
    """
    This set cover solver method returns 1 filed

      * `containing_set` is a minimal set of sets in which all the element have been contained

    The input arguments consist of

      * `element_vertexes` that is a list of element nodes
      * `set_vertexes` that is a list of set nodes those elements are contained in

    Element and set equivalents at a certain index come in pairs so mappings between sets and elements are consistent.

    The procedure can be invoked in openCypher using the following calls, e.g.:
      CALL set_cover.cp_solve([(:Point), (:Point)], [(:Set), (:Set)]) YIELD containing_set;

    The method uses constraint programming as a solving tool for obtaining a minimal set of sets that contain
        all the elements.
    """

    creator = GekkoMatchingProblemCreator()
    mp = creator.create_matching_problem(element_vertexes, set_vertexes)

    solver = GekkoMPSolver()
    result = solver.solve(matching_problem=mp)

    resulting_nodes = [context.graph.get_vertex_by_id(x) for x in result]

    computed_return_value = [mgp_Record(containing_set=x) for x in resulting_nodes]
    return computed_return_value


@mgp_read_proc
def greedy(
    context: mgp_ProcCtx,
    element_vertexes: list[mgp_Vertex],
    set_vertexes: list[mgp_Vertex],
) -> list[mgp_Record]:
    """
    This set cover solver method returns 1 filed

      * `containing_set` is one set of a cover in which all the elements have been contained

    The input arguments consist of

      * `element_vertexes` that is a list of element nodes
      * `set_vertexes` that is a list of set nodes those elements are contained in

    Element and set equivalents at a certain index come in pairs so mappings between sets and elements are consistent.

    The procedure can be invoked in openCypher using the following calls, e.g.:
      CALL set_cover.greedy([(:Point), (:Point)], [(:Set), (:Set)]) YIELD containing_set;

    The method repeatedly picks the set covering the most still-uncovered elements. The cover is within the greedy
        H(n) <= ln(n) + 1 factor of the minimum (n = largest set size) but is not guaranteed minimal; use cp_solve
        for an exact minimum.
    """

    creator = GreedyMatchingProblemCreator()
    mp = creator.create_matching_problem(element_vertexes, set_vertexes)

    solver = GreedyMPSolver()
    result = solver.solve(matching_problem=mp)

    resulting_nodes = [context.graph.get_vertex_by_id(x) for x in result]

    computed_return_value = [mgp_Record(containing_set=x) for x in resulting_nodes]
    return computed_return_value


def paired_membership_ids(element_vertexes: list[mgp_Vertex], set_vertexes: list[mgp_Vertex]) -> tuple[list[int], list[int]]:
    """
    Admits the paired element/set membership lists. Each index pairs one element with one containing set, so the
    lists must have equal length; every declared element is then represented by at least one membership.
    """
    if len(element_vertexes) != len(set_vertexes):
        raise ValueError(
            f"element_vertexes and set_vertexes must be paired: received {len(element_vertexes)} elements "
            f"and {len(set_vertexes)} sets"
        )
    element_values = [x.id for x in element_vertexes]
    set_values = [x.id for x in set_vertexes]
    computed_return_value = (element_values, set_values)
    return computed_return_value


class MatchingProblemCreator(abc_ABC):
    """
    Creator abstract class of matching problems
    """

    @abc_abstractmethod
    def create_matching_problem(self, element_vertexes, set_vertexes):
        """
        Creates a matching problem
        :param element_vertexes: Element vertexes pair component list
        :param set_vertexes: Set vertexes pair component list
        :return: matching problem
        """

        ...


class GreedyMatchingProblemCreator(MatchingProblemCreator):
    """
    Creator class for set cover to be solved with greedy method
    """

    def create_matching_problem(self, element_vertexes: list[mgp_Vertex], set_vertexes: list[mgp_Vertex]):
        """
        Creates a matching problem to be solved with greedy method
        :param element_vertexes: Element vertexes pair component list
        :param set_vertexes: Set vertexes pair component list
        :return: matching problem
        """

        element_values, set_values = paired_membership_ids(element_vertexes, set_vertexes)
        all_elements = set(element_values)
        all_sets = set(set_values)

        elements_by_sets: dict[int, set[int]] = {}
        for element, contained_set in zip(element_values, set_values, strict=True):
            set_elements = elements_by_sets.get(contained_set, set())
            set_elements.add(element)
            elements_by_sets[contained_set] = set_elements

        computed_return_value = GreedyMatchingProblem(all_elements, all_sets, elements_by_sets)
        return computed_return_value


class GekkoMatchingProblemCreator(MatchingProblemCreator):
    """
    Creator class for set cover to be solved with gekko constraint programming
    """

    def create_matching_problem(self, element_vertexes: list[mgp_Vertex], set_vertexes: list[mgp_Vertex]):
        """
        Creates a matching problem to be solved with gekko constraint programming method
        :param element_vertexes: Element vertexes pair component list
        :param set_vertexes: Set vertexes pair component list
        :return: matching problem
        """

        element_values, set_values = paired_membership_ids(element_vertexes, set_vertexes)
        set_values_distinct = set(set_values)

        sets_by_elements: dict[int, set[int]] = {}
        for element, contained_set in zip(element_values, set_values, strict=True):
            element_sets = sets_by_elements.get(element, set())
            element_sets.add(contained_set)
            sets_by_elements[element] = element_sets

        computed_return_value = GekkoMatchingProblem(set_values_distinct, sets_by_elements)
        return computed_return_value
