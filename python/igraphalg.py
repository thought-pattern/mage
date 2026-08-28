"""Utilities for igraphalg."""

from collections import defaultdict
from typing import Dict, List

from mgp import List as mgp_List
from mgp import Nullable as mgp_Nullable
from mgp import Number as mgp_Number
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from mgp_igraph import (
    CommunityDetectionObjectiveFunctionOptions,
    InvalidCommunityDetectionObjectiveFunctionException,
    InvalidPageRankImplementationOption,
    InvalidTopologicalSortingModeException,
    MemgraphIgraph,
    PageRankImplementationOptions,
    TopologicalSortException,
    TopologicalSortingModes,
)


@mgp_read_proc
def maxflow(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    capacity: str = "weight",
) -> mgp_Record(max_flow=mgp_Number):
    graph = MemgraphIgraph(ctx=ctx, directed=True)
    max_flow_value = graph.maxflow(source=source, target=target, capacity=capacity)

    _return_value = mgp_Record(max_flow=max_flow_value)
    return _return_value


@mgp_read_proc
def pagerank(
    ctx: mgp_ProcCtx,
    damping: mgp_Number = 0.85,
    weights: mgp_Nullable[str] = False,
    directed: bool = True,
    implementation: str = "prpack",
) -> mgp_Record(node=mgp_Vertex, rank=float):
    if implementation not in [
        PageRankImplementationOptions.PRPACK.value,
        PageRankImplementationOptions.ARPACK.value,
    ]:
        raise InvalidPageRankImplementationOption(
            'Implementation argument value can be "prpack" or "arpack"'
        )
    graph = MemgraphIgraph(ctx=ctx, directed=directed)
    pagerank_values = graph.pagerank(
        weights=weights,
        directed=directed,
        damping=damping,
        implementation=implementation,
    )

    _return_value = [mgp_Record(node=node, rank=rank) for node, rank in pagerank_values]
    return _return_value


@mgp_read_proc
def get_all_simple_paths(
    ctx: mgp_ProcCtx,
    v: mgp_Vertex,
    to: mgp_Vertex,
    cutoff: int = -1,
) -> mgp_Record(path=mgp_List[mgp_Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=True)

    _return_value = [
        mgp_Record(path=path)
        for path in graph.get_all_simple_paths(v=v, to=to, cutoff=cutoff)
    ]
    return _return_value


@mgp_read_proc
def mincut(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    capacity: mgp_Nullable[str] = False,
    directed: bool = True,
) -> mgp_Record(node=mgp_Vertex, partition_id=int):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    partition_vertices, _ = graph.mincut(
        source=source, target=target, capacity=capacity
    )

    _return_value = [
        mgp_Record(node=node, partition_id=i)
        for i, partition_nodes in enumerate(partition_vertices)
        for node in partition_nodes
    ]
    return _return_value


@mgp_read_proc
def topological_sort(ctx: mgp_ProcCtx, mode: str = "out") -> mgp_Record(
    nodes=mgp_List[mgp_Vertex]
):
    if mode not in [
        TopologicalSortingModes.IN.value,
        TopologicalSortingModes.OUT.value,
    ]:
        raise InvalidTopologicalSortingModeException(
            'Mode can only be either "out" or "in"'
        )
    if contains_cycle(ctx):
        raise TopologicalSortException(
            "Topological sort can't be performed on graph that contains cycle!"
        )

    graph = MemgraphIgraph(ctx=ctx, directed=True)
    sorted_nodes = graph.topological_sort(mode=mode)

    _return_value = mgp_Record(
        nodes=sorted_nodes,
    )
    return _return_value


@mgp_read_proc
def community_leiden(
    ctx: mgp_ProcCtx,
    objective_function: str = "CPM",
    weights: mgp_Nullable[str] = False,
    resolution_parameter: float = 1.0,
    beta: float = 0.01,
    initial_membership: mgp_Nullable[mgp_Nullable[List[mgp_Nullable[int]]]] = False,
    n_iterations: int = 2,
    node_weights: mgp_Nullable[List[mgp_Nullable[float]]] = False,
) -> mgp_Record(node=mgp_Vertex, community_id=int):
    if objective_function not in [
        CommunityDetectionObjectiveFunctionOptions.CPM.value,
        CommunityDetectionObjectiveFunctionOptions.MODULARITY.value,
    ]:
        raise InvalidCommunityDetectionObjectiveFunctionException(
            'Objective function can only be "CPM" or "modularity"'
        )

    graph = MemgraphIgraph(ctx=ctx, directed=False)

    communities = graph.community_leiden(
        resolution_parameter=resolution_parameter,
        weights=weights,
        n_iterations=n_iterations,
        objective_function=objective_function,
        beta=beta,
        initial_membership=initial_membership,
        node_weights=node_weights,
    )

    _return_value = [
        mgp_Record(
            node=node,
            community_id=community_id,
        )
        for node, community_id in communities
    ]
    return _return_value


@mgp_read_proc
def spanning_tree(
    ctx: mgp_ProcCtx, weights: mgp_Nullable[str] = False, directed: bool = False
) -> mgp_Record(tree=List[List[mgp_Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    _return_value = mgp_Record(tree=graph.spanning_tree(weights=weights))
    return _return_value


@mgp_read_proc
def shortest_path_length(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    weights: mgp_Nullable[str] = False,
    directed: bool = True,
) -> mgp_Record(length=float):
    graph = MemgraphIgraph(ctx, directed=directed)
    _return_value = mgp_Record(
        length=graph.shortest_path_length(
            source=source,
            target=target,
            weights=weights,
        )
    )
    return _return_value


@mgp_read_proc
def all_shortest_path_lengths(
    ctx: mgp_ProcCtx,
    weights: mgp_Nullable[str] = False,
    directed: bool = False,
) -> mgp_Record(src_node=mgp_Vertex, dest_node=mgp_Vertex, length=float):
    graph = MemgraphIgraph(ctx, directed=directed)
    lengths = graph.all_shortest_path_lengths(weights=weights)

    _return_value = [
        mgp_Record(
            src_node=graph.get_vertex_by_id(i),
            dest_node=graph.get_vertex_by_id(j),
            length=float(lengths[i][j]),
        )
        for i in range(len(lengths))
        for j in range(len(lengths[i]))
    ]
    return _return_value


@mgp_read_proc
def get_shortest_path(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    weights: mgp_Nullable[str] = False,
    directed: bool = True,
) -> mgp_Record(path=List[mgp_Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    _return_value = mgp_Record(
        path=graph.get_shortest_path(source=source, target=target, weights=weights)
    )
    return _return_value


def dfs(node: mgp_Vertex, visited: Dict[int, bool], stack: Dict[int, bool]) -> bool:
    """Depth-first-search algorithm with modification.

    Args:
        node (mgp.Vertex): Current node.
        visited (Dict[int,bool]): Dictionary with all nodes id that we visited.
        stack (Dict[int,bool]): Dictionary with nodes id that we encountered while traversing a node.

    Returns:
        bool: True if there is cycle else False.
    """

    visited[node.id] = True
    stack[node.id] = True

    for edge in node.out_edges:
        neighbour = edge.to_vertex
        if not visited.get(neighbour.id, False):
            if dfs(neighbour, visited, stack):
                return True
        elif stack.get(neighbour.id, False):
            return True

    stack[node.id] = False
    return False


def contains_cycle(ctx: mgp_ProcCtx) -> bool:
    """Method for checking if graph contains a cycle.

    Args:
        ctx (mgp.ProcCtx): Graph

    Returns:
        bool: True if there is cycle else False
    """

    visited, stack = defaultdict(bool), defaultdict(bool)
    for node in ctx.graph.vertices:
        if not visited.get(node.id, False):
            if dfs(node, visited, stack):
                return True
    return False
