"""Utilities for vrp."""

from mgp import Nullable as mgp_Nullable
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import Vertices as mgp_Vertices
from mgp import read_proc as mgp_read_proc
from numpy import ndarray as np_ndarray

from mage.constraint_programming import VRPConstraintProgrammingSolver
from mage.geography import (
    LATITUDE,
    LONGITUDE,
    create_distance_matrix,
)

internal_distance_matrix: object = False
internal_depot_index = -1

MAX_DISTANCE_MATRIX_SIZE = 100


def get_distance_matrix(vertices):
    """
    Assigns distance matrix global object or returns if its already there.
    """
    global internal_distance_matrix

    if isinstance(internal_distance_matrix, np_ndarray):
        return internal_distance_matrix

    vertex_positions: list[dict[str, float]] = []
    for vertex in vertices:
        vertex_positions.append(
            {
                LATITUDE: vertex.properties.get(LATITUDE, False),
                LONGITUDE: vertex.properties.get(LONGITUDE, False),
            }
        )

    computed_matrix = create_distance_matrix(vertex_positions)
    if not isinstance(computed_matrix, np_ndarray):
        raise ValueError("Unable to calculate a numeric distance matrix")
    internal_distance_matrix = computed_matrix

    return internal_distance_matrix


def get_depot_index(vertices: mgp_Vertices, depot_node: mgp_Vertex):
    """
    Assigns depot index global variable or returns if its already there.
    """
    global internal_depot_index

    if internal_depot_index >= 0:
        return internal_depot_index

    for i, vertex in enumerate(vertices):
        if vertex == depot_node:
            internal_depot_index = i
            break

    if internal_depot_index < 0:
        raise DepotUnspecifiedException("No depot location specified!")

    return internal_depot_index


def cleanup():
    global internal_distance_matrix, internal_depot_index

    distance_matrix = internal_distance_matrix
    if isinstance(distance_matrix, np_ndarray) and len(distance_matrix) >= MAX_DISTANCE_MATRIX_SIZE:
        internal_distance_matrix = False
        internal_depot_index = -1
    return False


@mgp_read_proc
def route(
    context: mgp_ProcCtx,
    depot_node: mgp_Vertex,
    number_of_vehicles: mgp_Nullable[int] = False,
) -> list[mgp_Record]:
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

    solver = VRPConstraintProgrammingSolver(number_of_vehicles, distance_matrix, depot_index)
    solver.solve()

    result = solver.get_result()

    cleanup()

    computed_return_value = [
        mgp_Record(from_vertex=vertices[x.from_vertex], to_vertex=vertices[x.to_vertex]) for x in result.vrp_paths
    ]
    return computed_return_value


class DepotUnspecifiedException(Exception):
    pass
