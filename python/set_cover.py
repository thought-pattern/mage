"""Utilities for set cover."""

from mgp import ProcCtx as mgp_ProcCtx, Record as mgp_Record, Vertex as mgp_Vertex, read_proc as mgp_read_proc

from mage.constraint_programming.solver import (
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
) -> mgp_Record(containing_set=mgp_Vertex):
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

    element_values, set_values = paired_membership_ids(element_vertexes, set_vertexes)
    mp = GekkoMatchingProblem(set(set_values), sets_by_element(element_values, set_values))

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
) -> mgp_Record(containing_set=mgp_Vertex):
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

    element_values, set_values = paired_membership_ids(element_vertexes, set_vertexes)
    mp = GreedyMatchingProblem(set(element_values), set(set_values), elements_by_set(element_values, set_values))

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


def elements_by_set(element_values: list[int], set_values: list[int]) -> dict[int, set[int]]:
    """Groups the paired memberships by containing set: each set id maps to the element ids it contains."""
    grouped: dict[int, set[int]] = {}
    for element, contained_set in zip(element_values, set_values, strict=True):
        set_elements = grouped.get(contained_set, set())
        set_elements.add(element)
        grouped[contained_set] = set_elements
    return grouped


def sets_by_element(element_values: list[int], set_values: list[int]) -> dict[int, set[int]]:
    """Groups the paired memberships by element: each element id maps to the set ids that contain it."""
    grouped: dict[int, set[int]] = {}
    for element, contained_set in zip(element_values, set_values, strict=True):
        element_sets = grouped.get(element, set())
        element_sets.add(contained_set)
        grouped[element] = element_sets
    return grouped
