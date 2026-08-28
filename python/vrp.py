"""Utilities for vrp."""

from typing import Dict, List

from mgp import Nullable as mgp_Nullable
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import Vertices as mgp_Vertices
from mgp import read_proc as mgp_read_proc

from mage.constraint_programming import VRPConstraintProgrammingSolver
from mage.geography import (
    LATITUDE,
    LONGITUDE,
    create_distance_matrix,
)

__distance_matrix = False
__depot_index = -1

MAX_DISTANCE_MATRIX_SIZE = 100


def get_distance_matrix(vertices):
    """
    Assigns distance matrix global object or returns if its already there.
    """
    global __distance_matrix

    if __distance_matrix is not False:
        return __distance_matrix

    vertex_positions: List[Dict[str, float]] = []
    for vertex in vertices:
        vertex_positions.append(
            {
                LATITUDE: vertex.properties.get(LATITUDE, False),
                LONGITUDE: vertex.properties.get(LONGITUDE, False),
            }
        )

    __distance_matrix = create_distance_matrix(vertex_positions)

    return __distance_matrix


def get_depot_index(vertices: mgp_Vertices, depot_node: mgp_Vertex):
    """
    Assigns depot index global variable or returns if its already there.
    """
    global __depot_index

    if __depot_index >= 0:
        return __depot_index

    for i, vertex in enumerate(vertices):
        if vertex == depot_node:
            __depot_index = i
            break

    if __depot_index < 0:
        raise DepotUnspecifiedException("No depot location specified!")

    return __depot_index


def cleanup():
    global __distance_matrix, __depot_index

    if (
        __distance_matrix is not False
        and len(__distance_matrix) >= MAX_DISTANCE_MATRIX_SIZE
    ):
        __distance_matrix = False
        __depot_index = -1
    return False


@mgp_read_proc
def route(
    context: mgp_ProcCtx,
    depot_node: mgp_Vertex,
    number_of_vehicles: mgp_Nullable[int] = False,
) -> mgp_Record(from_vertex=mgp_Vertex, to_vertex=mgp_Vertex):
    """
    The VRP routing returns 2 fields.
        * `from_vertex` represents the starting nodes out of all selected routes (edges) in the complete graph
        * `to_vertex` represents the ending nodes out of all selected routes (edges) in the complete graph

    The input arguments are:
        * `number_of_vehicle` represents the cardinality of fleet with which the problem is going to be solved
        * `depot_label` represents the name of the label which contains the depot node
    """

    if number_of_vehicles is None:
        number_of_vehicles = False
    if number_of_vehicles is False:
        number_of_vehicles = 1
    if number_of_vehicles <= 0:
        raise Exception("Number of vehicles must be greater than 0.")

    vertices = [v for v in context.graph.vertices]
    distance_matrix = get_distance_matrix(vertices)
    depot_index = get_depot_index(vertices, depot_node)

    solver = VRPConstraintProgrammingSolver(
        number_of_vehicles, distance_matrix, depot_index
    )
    solver.solve()

    result = solver.get_result()

    cleanup()

    _return_value = [
        mgp_Record(from_vertex=vertices[x.from_vertex], to_vertex=vertices[x.to_vertex])
        for x in result.vrp_paths
    ]
    return _return_value


class DepotUnspecifiedException(Exception):
    pass
