"""Utilities for graph coloring."""

from mgp import Edge as mgp_Edge
from mgp import List as mgp_List
from mgp import Map as mgp_Map
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc

from mage.graph_coloring_module import Graph, Parameter
from mage.graph_coloring_module.parameter_boundary import normalize_parameters

DEFAULT_ARGUMENT_DICT = {}


@mgp_read_proc
def color_graph(
    context: mgp_ProcCtx,
    parameters: mgp_Map = DEFAULT_ARGUMENT_DICT,
    edge_property: str = "weight",
) -> list[mgp_Record]:
    """
    Example:
    CALL graph_coloring.color_graph() YIELD *;
    """
    if parameters is DEFAULT_ARGUMENT_DICT:
        parameters = DEFAULT_ARGUMENT_DICT.copy()
    parameters = normalize_parameters(dict(parameters))
    nodes, adj_list = graph_adjacency(context, edge_property)
    graph = Graph(nodes, adj_list)
    algorithm = parameters.get(Parameter.ALGORITHM, False)
    solution = algorithm.run(graph, parameters, abort_check=context.check_must_abort)
    computed_return_value = [
        mgp_Record(node=context.graph.get_vertex_by_id(graph.label(node)), color=color)
        for node, color in enumerate(solution.chromosome)
    ]
    return computed_return_value


@mgp_read_proc
def color_subgraph(
    context: mgp_ProcCtx,
    vertices: mgp_List[mgp_Vertex],
    edges: mgp_List[mgp_Edge],
    parameters: mgp_Map = DEFAULT_ARGUMENT_DICT,
    edge_property: str = "weight",
) -> list[mgp_Record]:
    """
    Example:
    MATCH (a:Cell)-[e:CLOSE_TO]->(b:Cell)
    WITH collect(a) as nodes, collect (e) as edges
    CALL graph_coloring.color_subgraph(nodes, edges, {no_of_colors: 2})
    YIELD color, node
    RETURN color, node;
    """
    if parameters is DEFAULT_ARGUMENT_DICT:
        parameters = DEFAULT_ARGUMENT_DICT.copy()
    parameters = normalize_parameters(dict(parameters))
    nodes, adj_list = subgraph_adjacency(context, vertices, edges, edge_property)
    graph = Graph(nodes, adj_list)
    algorithm = parameters.get(Parameter.ALGORITHM, False)
    solution = algorithm.run(graph, parameters, abort_check=context.check_must_abort)
    computed_return_value = [
        mgp_Record(node=context.graph.get_vertex_by_id(graph.label(node)), color=color)
        for node, color in enumerate(solution.chromosome)
    ]
    return computed_return_value


def graph_adjacency(context: mgp_ProcCtx, edge_property: str) -> tuple[list, dict]:
    """Returns the graph's vertex ids and undirected weighted adjacency (edge_property, default weight 1)."""
    nodes = []
    adj_list = {}

    for v in context.graph.vertices:
        context.check_must_abort()
        nodes.append(v.id)

    for v in context.graph.vertices:
        context.check_must_abort()
        for e in v.out_edges:
            weight = e.properties.get(edge_property, 1)
            adj_list.setdefault(e.from_vertex.id, []).append((e.to_vertex.id, weight))
            adj_list.setdefault(e.to_vertex.id, []).append((e.from_vertex.id, weight))

    adjacency = (nodes, adj_list)
    return adjacency


def subgraph_adjacency(
    context: mgp_ProcCtx,
    vertices: mgp_List[mgp_Vertex],
    edges: mgp_List[mgp_Edge],
    edge_property: str,
) -> tuple[list, dict]:
    """Returns the subgraph's vertex ids (edge endpoints included) and its undirected weighted adjacency."""
    vertices, edges = map(set, [vertices, edges])

    nodes = []
    adj_list = {}

    for v in vertices:
        context.check_must_abort()
        nodes.append(v.id)

    for e in edges:
        context.check_must_abort()
        weight = e.properties.get(edge_property, 1)
        if e.from_vertex.id not in nodes:
            nodes.append(e.from_vertex.id)
        if e.to_vertex.id not in nodes:
            nodes.append(e.to_vertex.id)
        adj_list.setdefault(e.from_vertex.id, []).append((e.to_vertex.id, weight))
        adj_list.setdefault(e.to_vertex.id, []).append((e.from_vertex.id, weight))

    adjacency = (nodes, adj_list)
    return adjacency
