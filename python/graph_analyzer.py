"""Utilities for graph analyzer."""

from collections import OrderedDict
from inspect import cleandoc
from itertools import chain, repeat
from sys import stderr as sys_stderr
from sys import version as sys_version
from typing import List, Tuple

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
        "\nNOTE: Please install networkx to be able to use graph_analyzer module. Using Python:\n"
        + sys_version
        + "\n"
    )
    raise import_error from import_error


_MAX_LIST_SIZE = 10


@mgp_read_proc
def help() -> mgp_Record(name=str, value=str):
    """Shows manual page for graph_analyzer."""
    records = []

    def make_records(name, doc):
        _return_value = (
            mgp_Record(name=n, value=v)
            for n, v in zip(
                chain([name], repeat("")), cleandoc(doc).splitlines(), strict=False
            )
        )
        return _return_value

    for func in (help, analyze, analyze_subgraph):
        records.extend(
            make_records("Procedure '{}'".format(func.__name__), func.__doc__)
        )

    for m, v in _get_analysis_mapping().items():
        records.extend(make_records("Analysis '{}'".format(m), v.__doc__))

    return records


@mgp_read_proc
def analyze(
    context: mgp_ProcCtx, analyses: mgp_Nullable[List[str]] = False
) -> mgp_Record(name=str, value=str):
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
    recs = _analyze_graph(context, g, analyses)
    _return_value = [mgp_Record(name=name, value=value) for name, value in recs]
    return _return_value


@mgp_read_proc
def analyze_subgraph(
    context: mgp_ProcCtx,
    vertices: mgp_List[mgp_Vertex],
    edges: mgp_List[mgp_Edge],
    analyses: mgp_Nullable[List[str]] = False,
) -> mgp_Record(name=str, value=str):
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
        lambda n: n in vertices,
        lambda n1, n2, e: e in edges,
    )
    recs = _analyze_graph(context, g, analyses)
    _return_value = [mgp_Record(name=name, value=value) for name, value in recs]
    return _return_value


def _get_analysis_mapping():
    _return_value = OrderedDict(
        [
            ("nodes", _number_of_nodes),
            ("edges", _number_of_edges),
            ("bridges", _bridges),
            ("articulation_points", _articulation_points),
            ("avg_degree", _avg_degree),
            ("sorted_nodes_degree", _sorted_nodes_degree),
            ("self_loops", _self_loops),
            ("is_bipartite", _is_bipartite),
            ("is_planar", _is_planar),
            ("is_biconnected: ", _is_biconnected),
            ("is_weakly_connected", _is_weakly_connected),
            ("number_of_weakly_components", _weakly_components),
            ("is_strongly_connected", _is_strongly_connected),
            ("strongly_components", _strongly_components),
            ("is_dag", _is_dag),
            ("is_eulerian", _is_eulerian),
            ("is_forest", _is_forest),
            ("is_tree", _is_tree),
        ]
    )
    return _return_value


def _get_analysis_func(name: str):
    _name_to_proc = _get_analysis_mapping()
    _return_value = _name_to_proc.get(name.lower(), False)
    return _return_value


def _get_analysis_funcs():
    _return_value = _get_analysis_mapping().values()
    return _return_value


def _analyze_graph(
    context: mgp_ProcCtx, g: nx_MultiDiGraph, analyses: List[str]
) -> List[Tuple[str, str]]:
    functions = (
        _get_analysis_funcs()
        if analyses is None
        else [_get_analysis_func(name) for name in analyses]
    )

    records = []
    for index, f in enumerate(functions):
        context.check_must_abort()
        if f is None:
            raise KeyError("Graph analysis is not supported: " + analyses[index])
        name, value = f(g)
        if isinstance(value, (list, set, tuple)):
            value = list(value)[:_MAX_LIST_SIZE]
        records.append((name, str(value)))

    return records


def _number_of_nodes(g: nx_MultiDiGraph) -> Tuple[str, int]:
    """Returns number of nodes."""
    _return_value = "Number of nodes", nx_number_of_nodes(g)
    return _return_value


def _number_of_edges(g: nx_MultiDiGraph) -> Tuple[str, int]:
    """Returns number of edges."""
    _return_value = "Number of edges", nx_number_of_edges(g)
    return _return_value


def _avg_degree(g: nx_MultiDiGraph) -> Tuple[str, float]:
    """Returns average degree."""
    _, number_of_nodes = _number_of_nodes(g)
    _, number_of_edges = _number_of_edges(g)
    avg_degree = 0 if number_of_nodes == 0 else number_of_edges / number_of_nodes
    return "Average degree", avg_degree


def _sorted_nodes_degree(g: nx_MultiDiGraph) -> Tuple[str, List[int]]:
    """Returns list of sorted nodes degree. [(node_id, degree), ...]"""
    nodes_degree = [(n, g.degree(n)) for n in g.nodes()]
    nodes_degree.sort(key=lambda x: x[1], reverse=True)
    return "Sorted nodes degree", nodes_degree


def _self_loops(g: nx_MultiDiGraph) -> Tuple[str, int]:
    """Returns number of self loops."""
    _return_value = "Self loops", sum(1 if e[0] == e[1] else 0 for e in g.edges())
    return _return_value


def _is_bipartite(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Checks if graph is bipartite."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = (
        False if number_of_nodes == 0 else nx_algorithms.bipartite.basic.is_bipartite(g)
    )
    return "Is bipartite", ret


def _is_planar(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Checks if graph is planar."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = (
        False if number_of_nodes == 0 else nx_algorithms.planarity.check_planarity(g)[0]
    )
    return "Is planar", ret


def _is_biconnected(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Check if graph is biconnected."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = (
        False
        if number_of_nodes == 0
        else nx_is_biconnected(nx_MultiDiGraph.to_undirected(g))
    )
    return "Is biconnected", ret


def _is_weakly_connected(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Check if graph is weakly connected."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_is_weakly_connected(g)
    return "Is weakly connected", ret


def _is_strongly_connected(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Checks if graph is strongly connected."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_is_strongly_connected(g)
    return "Is strongly connected", ret


def _is_dag(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Check if graph is directed acyclic graph (DAG)"""
    _, number_of_nodes = _number_of_nodes(g)
    ret = (
        False
        if number_of_nodes == 0
        else nx_algorithms.dag.is_directed_acyclic_graph(g)
    )
    return "Is DAG", ret


def _is_eulerian(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Checks if graph is Eulerian."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.euler.is_eulerian(g)
    return "Is eulerian", ret


def _is_forest(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Checks if graph is forest, all components must be trees."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.tree.recognition.is_forest(g)
    return "Is forest", ret


def _is_tree(g: nx_MultiDiGraph) -> Tuple[str, bool]:
    """Checks if graph is tree."""
    _, number_of_nodes = _number_of_nodes(g)
    ret = False if number_of_nodes == 0 else nx_algorithms.tree.recognition.is_tree(g)
    return "Is tree", ret


def _bridges(g: nx_MultiDiGraph) -> Tuple[str, int]:
    """Returns number of bridges, multiple edges between same nodes are
    mapped to one edge."""
    _return_value = "Number of bridges", sum(1 for _ in nx_bridges(nx_Graph(g)))
    return _return_value


def _articulation_points(g: nx_MultiDiGraph):
    """Returns number of articulation points."""
    undirected = nx_MultiDiGraph.to_undirected(g)
    _return_value = (
        "Number of articulation points",
        sum(1 for _ in nx_articulation_points(undirected)),
    )
    return _return_value


def _weakly_components(g: nx_MultiDiGraph):
    """Returns number of weakly components."""
    comps = nx_algorithms.components.number_weakly_connected_components(g)
    return "Number of weakly connected components", comps


def _strongly_components(g: nx_MultiDiGraph):
    """Returns number of strongly connected components."""
    comps = nx_algorithms.components.number_strongly_connected_components(g)
    return "Number of strongly connected components", comps
