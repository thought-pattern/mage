"""Utilities for tsp."""

from mgp import (
    List as mgp_List,
    Nullable as mgp_Nullable,
    Record as mgp_Record,
    Vertex as mgp_Vertex,
    read_proc as mgp_read_proc,
)

from mage.geography.travelling_salesman import create_distance_matrix, solve_1_5_approx, solve_2_approx, solve_greedy

DEFAULT_SOLVING_METHOD = "1.5_approx"
tsp_solving_methods = {
    "2_approx": solve_2_approx,
    "greedy": solve_greedy,
    DEFAULT_SOLVING_METHOD: solve_1_5_approx,
}


def normalize_solving_method(method: str) -> str:
    """Normalizes documented TSP method names (case and hyphen/underscore) and rejects unknown methods."""
    if not isinstance(method, str):
        raise TypeError("TSP solving method must be a string.")
    normalized = method.strip().lower().replace("-", "_")
    if normalized not in tsp_solving_methods:
        raise ValueError(f"Unsupported TSP solving method {method!r}; choose '1.5-approx', '2-approx', or 'greedy'.")
    return normalized


def solve(points: list[mgp_Vertex], method: str = DEFAULT_SOLVING_METHOD) -> mgp_Record:
    """
    The tsp solver returns 2 fields whose elements at indexes are correlated

      * `sources` - elements from 1st to n-1th element
      * `destinations` - elements from 2nd to nth element

    The pairs of them represent individual edges between 2 nodes in the graph.

    The required argument is the list of cities one wants to find the path from.
    The optional argument `method` is by default '1.5-approx'. Other arguments that can be
    specified are '2-approx' and 'greedy'.

    The procedure can be invoked in openCypher using the following calls:
    MATCH (n:Point)
    WITH collect(n) as points
    CALL tsp.solve(points) YIELD sources, destinations RETURN sources, destinations;
    """

    if not all(isinstance(x, mgp_Vertex) for x in points):
        computed_return_value = mgp_Record(sources=None, destinations=None)
        return computed_return_value

    dm = create_distance_matrix([dict(x.properties.items()) for x in points])

    if dm is False:
        computed_return_value = mgp_Record(sources=None, destinations=None)
        return computed_return_value

    selected_method = normalize_solving_method(method)
    solver = tsp_solving_methods.get(selected_method, solve_1_5_approx)
    order = solver(dm)

    sources = [points[order[x]] for x in range(len(order) - 1)]
    destinations = [points[order[x]] for x in range(1, len(order))]

    computed_return_value = mgp_Record(sources=sources, destinations=destinations)
    return computed_return_value


solve.__annotations__["return"] = mgp_Record(
    sources=mgp_Nullable[mgp_List[mgp_Vertex]],
    destinations=mgp_Nullable[mgp_List[mgp_Vertex]],
)
solve = mgp_read_proc(solve)
