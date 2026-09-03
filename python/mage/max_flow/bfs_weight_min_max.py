"""Utilities for bfs weight min max."""

from collections import deque
from collections.abc import Iterable
from collections.abc import Mapping


def BFS_find_weight_min_max(start_v: object, edge_property: str) -> tuple[float, float]:
    """
    Breadth-first search for finding the largest and smallest edge weight,
    largest being used for capacity scaling, and smallest for lower bound

    :param start_v: starting vertex
    :param edge_property: str denoting the edge property used as weight

    :return: Number, the largest value of edge_property in graph
    """

    if not isinstance(edge_property, str) or not edge_property:
        raise ValueError("edge_property must be a nonempty string")
    if not hasattr(start_v, "out_edges"):
        raise TypeError("start_v must provide out_edges")
    next_queue = deque([start_v])
    visited: set[object] = set()
    max_weight = float("-Inf")
    min_weight = float("Inf")
    found_weight = False

    while next_queue:
        current_v = next_queue.popleft()
        visited.add(current_v)

        out_edges = getattr(current_v, "out_edges", False)
        if not isinstance(out_edges, Iterable):
            raise TypeError("every traversed vertex must provide iterable out_edges")
        for e in out_edges:
            # if there are edges without the given property, we ignore them,
            # in order to support heterogeneous graphs
            properties = getattr(e, "properties", False)
            if not isinstance(properties, Mapping):
                raise TypeError("every traversed edge must provide a property mapping")
            if edge_property not in properties:
                continue

            weight = properties.get(edge_property, False)
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise TypeError(f"edge property {edge_property!r} must be numeric")
            max_weight = max(max_weight, float(weight))
            min_weight = min(min_weight, float(weight))
            found_weight = True

            to_vertex = getattr(e, "to_vertex", False)
            if to_vertex is False:
                raise TypeError("every traversed edge must provide to_vertex")
            if to_vertex not in visited:
                next_queue.append(to_vertex)

    if not found_weight:
        raise ValueError(f"no reachable edge provides property {edge_property!r}")
    return max_weight, min_weight
