"""Utilities for vrp."""

from mgp import Nullable as mgp_Nullable
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from numpy import ndarray as np_ndarray

from mage.constraint_programming import VRPConstraintProgrammingSolver
from mage.geography import (
    LATITUDE,
    LONGITUDE,
    create_distance_matrix,
)


def get_distance_matrix(vertices: list[mgp_Vertex]) -> np_ndarray:
    """
    Computes the distance matrix for exactly this request's ordered vertex list. Matrix rows and columns are
    positions in `vertices`, so the matrix is never reused across calls whose vertices or coordinates may differ.
    """
    vertex_positions: list[dict[str, float]] = []
    for vertex in vertices:
        vertex_positions.append(
            {
                LATITUDE: vertex.properties.get(LATITUDE, False),
                LONGITUDE: vertex.properties.get(LONGITUDE, False),
            }
        )

    distance_matrix = create_distance_matrix(vertex_positions)
    if not isinstance(distance_matrix, np_ndarray):
        raise ValueError("Unable to calculate a numeric distance matrix")
    return distance_matrix


def get_depot_index(vertices: list[mgp_Vertex], depot_node: mgp_Vertex) -> int:
    """
    Returns the position of this request's depot in this request's ordered vertex list.
    """
    depot_index = -1
    for i, vertex in enumerate(vertices):
        if vertex == depot_node:
            depot_index = i
            break

    if depot_index < 0:
        raise DepotUnspecifiedException("No depot location specified!")

    return depot_index


@mgp_read_proc
def route(
    context: mgp_ProcCtx,
    depot_node: mgp_Vertex,
    number_of_vehicles: mgp_Nullable[int] = None,
) -> list[mgp_Record]:
    """
    The VRP routing returns 2 fields.
        * `from_vertex` represents the starting nodes out of all selected routes (edges) in the complete graph
        * `to_vertex` represents the ending nodes out of all selected routes (edges) in the complete graph

    The input arguments are:
        * `number_of_vehicle` represents the cardinality of fleet with which the problem is going to be solved
        * `depot_label` represents the name of the label which contains the depot node
    """

    # Cypher null (the only default Memgraph admits for a nullable argument) means one vehicle.
    if number_of_vehicles is None:
        number_of_vehicles = 1
    if number_of_vehicles <= 0:
        raise Exception("Number of vehicles must be greater than 0.")

    # The ordered vertex list, its distance matrix and the depot position are bound together for this call only;
    # result indexes are mapped back through the same list.
    vertices = [v for v in context.graph.vertices]
    distance_matrix = get_distance_matrix(vertices)
    depot_index = get_depot_index(vertices, depot_node)

    solver = VRPConstraintProgrammingSolver(number_of_vehicles, distance_matrix, depot_index)
    solver.solve()

    result = solver.get_result()

    computed_return_value = [
        mgp_Record(from_vertex=vertices[x.from_vertex], to_vertex=vertices[x.to_vertex]) for x in result.vrp_paths
    ]
    return computed_return_value


class DepotUnspecifiedException(Exception):
    pass
