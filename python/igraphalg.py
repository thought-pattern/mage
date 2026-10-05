"""Utilities for igraphalg."""

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

    computed_return_value = mgp_Record(max_flow=max_flow_value)
    return computed_return_value


@mgp_read_proc
def pagerank(
    ctx: mgp_ProcCtx,
    damping: mgp_Number = 0.85,
    weights: mgp_Nullable[str] = None,
    directed: bool = True,
    implementation: str = "prpack",
) -> mgp_Record(node=mgp_Vertex, rank=float):
    if implementation not in [
        PageRankImplementationOptions.PRPACK.value,
        PageRankImplementationOptions.ARPACK.value,
    ]:
        raise InvalidPageRankImplementationOption('Implementation argument value can be "prpack" or "arpack"')
    graph = MemgraphIgraph(ctx=ctx, directed=directed)
    pagerank_values = graph.pagerank(
        weights=weights,
        directed=directed,
        damping=damping,
        implementation=implementation,
    )

    computed_return_value = [mgp_Record(node=node, rank=rank) for node, rank in pagerank_values]
    return computed_return_value


@mgp_read_proc
def get_all_simple_paths(
    ctx: mgp_ProcCtx,
    v: mgp_Vertex,
    to: mgp_Vertex,
    cutoff: int = -1,
) -> mgp_Record(path=mgp_List[mgp_Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=True)

    computed_return_value = [mgp_Record(path=path) for path in graph.get_all_simple_paths(v=v, to=to, cutoff=cutoff)]
    return computed_return_value


@mgp_read_proc
def mincut(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    capacity: mgp_Nullable[str] = None,
    directed: bool = True,
) -> mgp_Record(node=mgp_Vertex, partition_id=int):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    partition_vertices, _ = graph.mincut(source=source, target=target, capacity=capacity)

    computed_return_value = [
        mgp_Record(node=node, partition_id=i) for i, partition_nodes in enumerate(partition_vertices) for node in partition_nodes
    ]
    return computed_return_value


@mgp_read_proc
def topological_sort(ctx: mgp_ProcCtx, mode: str = "out") -> mgp_Record(nodes=mgp_List[mgp_Vertex]):
    if mode not in [
        TopologicalSortingModes.IN.value,
        TopologicalSortingModes.OUT.value,
    ]:
        raise InvalidTopologicalSortingModeException('Mode can only be either "out" or "in"')
    graph = MemgraphIgraph(ctx=ctx, directed=True)
    # igraph's linear-time DAG check replaces a host-graph recursion that overflowed on long valid chains.
    if not graph.is_dag():
        raise TopologicalSortException("Topological sort can't be performed on graph that contains cycle!")
    sorted_nodes = graph.topological_sort(mode=mode)

    computed_return_value = mgp_Record(
        nodes=sorted_nodes,
    )
    return computed_return_value


@mgp_read_proc
def community_leiden(
    ctx: mgp_ProcCtx,
    objective_function: str = "CPM",
    weights: mgp_Nullable[str] = None,
    resolution_parameter: float = 1.0,
    beta: float = 0.01,
    initial_membership: mgp_Nullable[mgp_Nullable[list[mgp_Nullable[int]]]] = None,
    n_iterations: int = 2,
    node_weights: mgp_Nullable[list[mgp_Nullable[float]]] = None,
) -> mgp_Record(node=mgp_Vertex, community_id=int):
    if objective_function not in [
        CommunityDetectionObjectiveFunctionOptions.CPM.value,
        CommunityDetectionObjectiveFunctionOptions.MODULARITY.value,
    ]:
        raise InvalidCommunityDetectionObjectiveFunctionException('Objective function can only be "CPM" or "modularity"')

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

    computed_return_value = [
        mgp_Record(
            node=node,
            community_id=community_id,
        )
        for node, community_id in communities
    ]
    return computed_return_value


@mgp_read_proc
def spanning_tree(
    ctx: mgp_ProcCtx, weights: mgp_Nullable[str] = None, directed: bool = False
) -> mgp_Record(tree=mgp_List[mgp_List[mgp_Vertex]]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    computed_return_value = mgp_Record(tree=graph.spanning_tree(weights=weights))
    return computed_return_value


@mgp_read_proc
def shortest_path_length(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    weights: mgp_Nullable[str] = None,
    directed: bool = True,
) -> mgp_Record(length=float):
    graph = MemgraphIgraph(ctx, directed=directed)
    computed_return_value = mgp_Record(
        length=graph.shortest_path_length(
            source=source,
            target=target,
            weights=weights,
        )
    )
    return computed_return_value


@mgp_read_proc
def all_shortest_path_lengths(
    ctx: mgp_ProcCtx,
    weights: mgp_Nullable[str] = None,
    directed: bool = False,
) -> mgp_Record(src_node=mgp_Vertex, dest_node=mgp_Vertex, length=float):
    graph = MemgraphIgraph(ctx, directed=directed)
    lengths = graph.all_shortest_path_lengths(weights=weights)

    computed_return_value = [
        mgp_Record(
            src_node=graph.get_vertex_by_id(i),
            dest_node=graph.get_vertex_by_id(j),
            length=float(lengths[i][j]),
        )
        for i in range(len(lengths))
        for j in range(len(lengths[i]))
    ]
    return computed_return_value


@mgp_read_proc
def get_shortest_path(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    weights: mgp_Nullable[str] = None,
    directed: bool = True,
) -> mgp_Record(path=mgp_List[mgp_Vertex]):
    graph = MemgraphIgraph(ctx=ctx, directed=directed)

    computed_return_value = mgp_Record(path=graph.get_shortest_path(source=source, target=target, weights=weights))
    return computed_return_value
