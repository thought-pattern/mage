"""Utilities for union find."""

from enum import Enum
from itertools import product

from mgp import (
    ProcCtx as mgp_ProcCtx,
    Record as mgp_Record,
    Vertex as mgp_Vertex,
    read_proc as mgp_read_proc,
)

from mage.union_find.disjoint_set import DisjointSet

cached_components: dict[tuple[tuple[int, ...], tuple[tuple[int, int], ...]], dict[int, int]] = {}


def resolve_graph_components(ctx: mgp_ProcCtx, update: bool) -> dict[int, int]:
    """Builds or looks up the connected-component root map for ctx.graph."""
    vertices = list(ctx.graph.vertices)
    vertex_ids = [vertex.id for vertex in vertices]
    edge_pairs: list[tuple[int, int]] = []
    for vertex in vertices:
        for edge in vertex.out_edges:
            source_id = edge.from_vertex.id
            target_id = edge.to_vertex.id
            ordered_pair = (source_id, target_id) if source_id <= target_id else (target_id, source_id)
            edge_pairs.append(ordered_pair)
    signature = (tuple(sorted(vertex_ids)), tuple(sorted(edge_pairs)))
    if not update and signature in cached_components:
        cached = cached_components.get(signature, {})
        return cached
    local_disjoint_set = DisjointSet(node_ids=vertex_ids)
    for source_id, target_id in edge_pairs:
        local_disjoint_set.union(node1_id=source_id, node2_id=target_id)
    component_roots = {node_id: local_disjoint_set.find(node_id) for node_id in vertex_ids}
    cached_components[signature] = component_roots
    return component_roots


def pair_is_connected(component_roots: dict[int, int], node1_id: int, node2_id: int) -> bool:
    """Returns whether two vertices share a connected component in component_roots."""
    if node1_id not in component_roots:
        raise KeyError(f"unknown union-find node {node1_id}")
    if node2_id not in component_roots:
        raise KeyError(f"unknown union-find node {node2_id}")
    same_component = component_roots.get(node1_id, -1) == component_roots.get(node2_id, -2)
    return same_component


class Mode(Enum):
    """
    Valid values of the mode parameter
    """

    CARTESIAN = "cartesian"
    PAIRWISE = "pairwise"


def connected(
    ctx: mgp_ProcCtx,
    nodes1: object,
    nodes2: object,
    mode: str = "pairwise",
    update: bool = True,
) -> list:
    """
    Returns whether two nodes (or each pair in the product of two node lists) belong to the same connected component
    of the graph.
    :param nodes1: Node or tuple of nodes
    :type nodes1: Union[mgp.Vertex, Tuple[mgp.Vertex]]
    :param nodes2: Node or tuple of nodes
    :type nodes2: Union[mgp.Vertex, Tuple[mgp.Vertex]]
    :param mode: Mode of operation: `pairwise` for pairwise operation or `cartesian`
    for operating on the Cartesian product of given tuples. Default value is `pairwise`.
    :type mode: str
    :param update: Updates the disjoint set data structure used by the algorithm. Use if graph has been changed
    since this method's last call. Default value is `True`.
    :type update: str

    :return:
    :rtype: mgp.Record
    """

    component_roots = resolve_graph_components(ctx=ctx, update=update)

    if isinstance(nodes1, mgp_Vertex):
        nodes1 = (nodes1,)
    elif isinstance(nodes1, (list, tuple)):
        nodes1 = tuple(nodes1)
    else:
        raise TypeError("Invalid type of first argument.")

    if isinstance(nodes2, mgp_Vertex):
        nodes2 = (nodes2,)
    elif isinstance(nodes2, (list, tuple)):
        nodes2 = tuple(nodes2)
    else:
        raise TypeError("Invalid type of second argument.")

    if mode.lower() == Mode.PAIRWISE.value:
        if len(nodes1) != len(nodes2):
            raise ValueError("Incompatible lengths of given arguments.")

        computed_return_value = [
            mgp_Record(
                node1=node1,
                node2=node2,
                connected=pair_is_connected(component_roots, node1_id=node1.id, node2_id=node2.id),
            )
            for node1, node2 in zip(nodes1, nodes2, strict=False)
        ]
        return computed_return_value

    elif mode.lower() == Mode.CARTESIAN.value:
        computed_return_value = [
            mgp_Record(
                node1=node1,
                node2=node2,
                connected=pair_is_connected(component_roots, node1_id=node1.id, node2_id=node2.id),
            )
            for node1, node2 in product(nodes1, nodes2)
        ]
        return computed_return_value

    error_message = (
        f'Mode {mode} is invalid, please specify one of the following: "{Mode.PAIRWISE.value}", "' f'{Mode.CARTESIAN.value}".'
    )
    raise ValueError(error_message)


connected.__annotations__["return"] = mgp_Record(node1=mgp_Vertex, node2=mgp_Vertex, connected=bool)
connected = mgp_read_proc(connected)
