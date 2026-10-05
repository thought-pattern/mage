"""Utilities for graph analyzer."""

from heapq import nlargest
from inspect import cleandoc
from itertools import chain, repeat
from operator import itemgetter
from sys import stderr as sys_stderr
from sys import version as sys_version

from mgp import Edge as mgp_Edge
from mgp import List as mgp_List
from mgp import Nullable as mgp_Nullable
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc

# Imported last because it also depends on networkx.
from mgp_networkx import MemgraphMultiDiGraph

try:
    from networkx import Graph as nx_Graph
    from networkx import MultiDiGraph as nx_MultiDiGraph
    from networkx import algorithms as nx_algorithms
    from networkx import articulation_points as nx_articulation_points
    from networkx import bridges as nx_bridges
    from networkx import is_biconnected as nx_is_biconnected
    from networkx import is_strongly_connected as nx_is_strongly_connected
    from networkx import is_weakly_connected as nx_is_weakly_connected
    from networkx import number_of_edges as nx_number_of_edges
    from networkx import number_of_nodes as nx_number_of_nodes
    from networkx import subgraph_view as nx_subgraph_view
except ImportError as import_error:
    sys_stderr.write(
        "\nNOTE: Please install networkx to be able to use graph_analyzer module. Using Python:\n" + sys_version + "\n"
    )
    raise import_error from import_error


MAX_LIST_SIZE = 10


@mgp_read_proc
def help() -> list[mgp_Record]:
    """Shows manual page for graph_analyzer."""
    documented = [("Procedure '{}'".format(func.__name__), func.__doc__) for func in (help, analyze, analyze_subgraph)]
    documented.extend(("Analysis '{}'".format(name), analysis.__doc__) for name, analysis in ANALYSES.items())

    # Each documented item's title names its first manual line; its remaining lines carry an empty name.
    records = []
    for title, doc in documented:
        doc_text = doc if isinstance(doc, str) else ""
        lines = cleandoc(doc_text).splitlines()
        records.extend(mgp_Record(name=name, value=line) for name, line in zip(chain([title], repeat("")), lines, strict=False))

    return records


@mgp_read_proc
def analyze(context: mgp_ProcCtx, analyses: mgp_Nullable[list[str]] = None) -> list[mgp_Record]:
    """
    Shows graph information.

    In case of multiple results, only the first 10 will be shown.

    The optional parameter is a list of graph analyses to run.
    If NULL, all available analyses are run.

    Example call (give all information):
        CALL graph_analyzer.analyze() YIELD *;

    Example call (with parameter):
        CALL graph_analyzer.analyze(['nodes', 'edges']) YIELD *;
    """
    g = MemgraphMultiDiGraph(ctx=context)
    recs = analyze_graph(context, g, analyses)
    computed_return_value = [mgp_Record(name=name, value=value) for name, value in recs]
    return computed_return_value


@mgp_read_proc
def analyze_subgraph(
    context: mgp_ProcCtx,
    vertices: mgp_List[mgp_Vertex],
    edges: mgp_List[mgp_Edge],
    analyses: mgp_Nullable[list[str]] = None,
) -> list[mgp_Record]:
    """
    Shows subgraph information.

    In case of multiple results, only the first 10 will be shown.

    The optional parameter is a list of graph analyses to run.
    If NULL, all available analyses are run.

    Example call (give all information):
        MATCH (n)-[e]->(m) WITH
        collect(n) AS nodes,
        collect(e) AS edges
        CALL graph_analyzer.analyze_subgraph(nodes, edges) YIELD *
        RETURN name, value;

    Example call (with parameter):
        MATCH (n)-[e]->(m) WITH
        collect(n) AS nodes,
        collect(e) AS edges
        CALL graph_analyzer.analyze_subgraph(nodes, edges, ['nodes', 'edges'])
        YIELD *
        RETURN name, value;
    """
    vertices, edges = map(set, [vertices, edges])
    g = nx_subgraph_view(
        MemgraphMultiDiGraph(ctx=context),
        filter_node=lambda n: n in vertices,
        filter_edge=lambda n1, n2, e: e in edges,
    )
    recs = analyze_graph(context, g, analyses)
    computed_return_value = [mgp_Record(name=name, value=value) for name, value in recs]
    return computed_return_value


def analyze_graph(context: mgp_ProcCtx, g: nx_MultiDiGraph, analyses: object) -> list[tuple[str, str]]:
    requested_analyses = analyses if isinstance(analyses, list) else []
    # An unsupported name reads as False, which the loop below refuses with the requested spelling.
    if requested_analyses:
        functions = [ANALYSES.get(name.lower(), False) for name in requested_analyses]
    else:
        functions = list(ANALYSES.values())

    records = []
    for index, f in enumerate(functions):
        context.check_must_abort()
        if not callable(f):
            analysis_name = requested_analyses[index] if index < len(requested_analyses) else ""
            raise KeyError("Graph analysis is not supported: " + analysis_name)
        result = f(g)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("graph analysis must return a name and value pair")
        name, value = result
        if not isinstance(name, str):
            raise TypeError("graph analysis name must be a string")
        if isinstance(value, (list, set, tuple)):
            value = list(value)[:MAX_LIST_SIZE]
        records.append((name, str(value)))

    return records


def internal_number_of_nodes(g: nx_MultiDiGraph) -> tuple[str, int]:
    """Returns number of nodes."""
    computed_return_value = "Number of nodes", nx_number_of_nodes(g)
    return computed_return_value


def internal_number_of_edges(g: nx_MultiDiGraph) -> tuple[str, int]:
    """Returns number of edges."""
    computed_return_value = "Number of edges", nx_number_of_edges(g)
    return computed_return_value


def internal_avg_degree(g: nx_MultiDiGraph) -> tuple[str, float]:
    """Returns the mean total (in + out) degree, the convention sorted_nodes_degree reports per node."""
    _, number_of_nodes = internal_number_of_nodes(g)
    total_degree = sum(degree for _, degree in g.degree())
    avg_degree = 0 if number_of_nodes == 0 else total_degree / number_of_nodes
    return "Average degree", avg_degree


def internal_sorted_nodes_degree(g: nx_MultiDiGraph) -> tuple[str, list[tuple[object, object]]]:
    """Returns the MAX_LIST_SIZE highest total-degree nodes, descending. [(node_id, degree), ...]"""
    # nlargest equals a stable descending sort truncated to the shown prefix, so ties keep node order.
    nodes_degree = nlargest(MAX_LIST_SIZE, g.degree(), key=itemgetter(1))
    return "Sorted nodes degree", nodes_degree


def internal_self_loops(g: nx_MultiDiGraph) -> tuple[str, int]:
    """Returns number of self loops."""
    computed_return_value = "Self loops", sum(1 if e[0] == e[1] else 0 for e in g.edges())
    return computed_return_value


def internal_is_bipartite(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Checks if graph is bipartite."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.bipartite.is_bipartite(g)
    return "Is bipartite", ret


def internal_is_planar(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Checks if graph is planar."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.check_planarity(g)[0]
    return "Is planar", ret


def internal_is_biconnected(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Check if graph is biconnected."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_is_biconnected(nx_MultiDiGraph.to_undirected(g))
    return "Is biconnected", ret


def internal_is_weakly_connected(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Check if graph is weakly connected."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_is_weakly_connected(g)
    return "Is weakly connected", ret


def internal_is_strongly_connected(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Checks if graph is strongly connected."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_is_strongly_connected(g)
    return "Is strongly connected", ret


def internal_is_dag(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Check if graph is directed acyclic graph (DAG)"""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.is_directed_acyclic_graph(g)
    return "Is DAG", ret


def internal_is_eulerian(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Checks if graph is Eulerian."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.is_eulerian(g)
    return "Is eulerian", ret


def internal_is_forest(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Checks if graph is forest, all components must be trees."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.tree.recognition.is_forest(g)
    return "Is forest", ret


def internal_is_tree(g: nx_MultiDiGraph) -> tuple[str, bool]:
    """Checks if graph is tree."""
    _, number_of_nodes = internal_number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.tree.recognition.is_tree(g)
    return "Is tree", ret


def internal_bridges(g: nx_MultiDiGraph) -> tuple[str, int]:
    """Returns number of bridges, multiple edges between same nodes are
    mapped to one edge."""
    computed_return_value = "Number of bridges", sum(1 for _ in nx_bridges(nx_Graph(g)))
    return computed_return_value


def internal_articulation_points(g: nx_MultiDiGraph):
    """Returns number of articulation points."""
    undirected = nx_MultiDiGraph.to_undirected(g)
    computed_return_value = (
        "Number of articulation points",
        sum(1 for _ in nx_articulation_points(undirected)),
    )
    return computed_return_value


def weakly_components(g: nx_MultiDiGraph):
    """Returns number of weakly components."""
    comps = nx_algorithms.components.number_weakly_connected_components(g)
    return "Number of weakly connected components", comps


def internal_strongly_components(g: nx_MultiDiGraph):
    """Returns number of strongly connected components."""
    comps = nx_algorithms.components.number_strongly_connected_components(g)
    return "Number of strongly connected components", comps


# The supported analyses by public name, in the order help lists them and analyze runs them by default.
ANALYSES = {
    "nodes": internal_number_of_nodes,
    "edges": internal_number_of_edges,
    "bridges": internal_bridges,
    "articulation_points": internal_articulation_points,
    "avg_degree": internal_avg_degree,
    "sorted_nodes_degree": internal_sorted_nodes_degree,
    "self_loops": internal_self_loops,
    "is_bipartite": internal_is_bipartite,
    "is_planar": internal_is_planar,
    "is_biconnected": internal_is_biconnected,
    "is_weakly_connected": internal_is_weakly_connected,
    "number_of_weakly_components": weakly_components,
    "is_strongly_connected": internal_is_strongly_connected,
    "strongly_components": internal_strongly_components,
    "is_dag": internal_is_dag,
    "is_eulerian": internal_is_eulerian,
    "is_forest": internal_is_forest,
    "is_tree": internal_is_tree,
}
