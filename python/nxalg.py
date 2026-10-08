"""Utilities for nxalg."""

from importlib import import_module as imported_import_module
from sys import stderr as sys_stderr, version as sys_version

from mgp import (
    Edge as mgp_Edge,
    List as mgp_List,
    Nullable as mgp_Nullable,
    Number as mgp_Number,
    ProcCtx as mgp_ProcCtx,
    Record as mgp_Record,
    Vertex as mgp_Vertex,
    read_proc as mgp_read_proc,
)

# Imported last because it also depends on networkx.
from mgp_networkx import (
    MemgraphDiGraph,
    MemgraphMultiDiGraph,
    PropertiesDictionary,
)

try:
    from networkx import (
        DiGraph as nx_DiGraph,
        MultiDiGraph as nx_MultiDiGraph,
        NetworkXNoCycle as nx_NetworkXNoCycle,
        all_shortest_paths as nx_all_shortest_paths,
        all_simple_paths as nx_all_simple_paths,
        ancestors as nx_ancestors,
        betweenness_centrality as nx_betweenness_centrality,
        bfs_edges as nx_bfs_edges,
        bfs_predecessors as nx_bfs_predecessors,
        bfs_successors as nx_bfs_successors,
        bfs_tree as nx_bfs_tree,
        biconnected_components as nx_biconnected_components,
        bridges as nx_bridges,
        center as nx_center,
        chain_decomposition as nx_chain_decomposition,
        check_planarity as nx_check_planarity,
        clustering as nx_clustering,
        communicability as nx_communicability,
        community as nx_community,
        core_number as nx_core_number,
        degree_assortativity_coefficient as nx_degree_assortativity_coefficient,
        descendants as nx_descendants,
        dfs_postorder_nodes as nx_dfs_postorder_nodes,
        dfs_predecessors as nx_dfs_predecessors,
        dfs_preorder_nodes as nx_dfs_preorder_nodes,
        dfs_successors as nx_dfs_successors,
        dfs_tree as nx_dfs_tree,
        diameter as nx_diameter,
        dominance_frontiers as nx_dominance_frontiers,
        dominating_set as nx_dominating_set,
        edge_bfs as nx_edge_bfs,
        edge_dfs as nx_edge_dfs,
        find_cliques as nx_find_cliques,
        find_cycle as nx_find_cycle,
        flow_hierarchy as nx_flow_hierarchy,
        global_efficiency as nx_global_efficiency,
        greedy_color as nx_greedy_color,
        has_eulerian_path as nx_has_eulerian_path,
        has_path as nx_has_path,
        immediate_dominators as nx_immediate_dominators,
        is_arborescence as nx_is_arborescence,
        is_at_free as nx_is_at_free,
        is_bipartite as nx_is_bipartite,
        is_branching as nx_is_branching,
        is_chordal as nx_is_chordal,
        is_distance_regular as nx_is_distance_regular,
        is_edge_cover as nx_is_edge_cover,
        is_eulerian as nx_is_eulerian,
        is_forest as nx_is_forest,
        is_isolate as nx_is_isolate,
        is_isomorphic as nx_is_isomorphic,
        is_semieulerian as nx_is_semieulerian,
        is_simple_path as nx_is_simple_path,
        is_strongly_regular as nx_is_strongly_regular,
        is_tree as nx_is_tree,
        isolates as nx_isolates,
        jaccard_coefficient as nx_jaccard_coefficient,
        k_components as nx_k_components,
        k_edge_components as nx_k_edge_components,
        local_efficiency as nx_local_efficiency,
        lowest_common_ancestor as nx_lowest_common_ancestor,
        maximal_matching as nx_maximal_matching,
        minimum_spanning_tree as nx_minimum_spanning_tree,
        multi_source_dijkstra_path as nx_multi_source_dijkstra_path,
        multi_source_dijkstra_path_length as nx_multi_source_dijkstra_path_length,
        node_boundary as nx_node_boundary,
        node_connectivity as nx_node_connectivity,
        node_expansion as nx_node_expansion,
        non_randomness as nx_non_randomness,
        pagerank as nx_pagerank,
        reciprocity as nx_reciprocity,
        shortest_path as nx_shortest_path,
        shortest_path_length as nx_shortest_path_length,
        simple_cycles as nx_simple_cycles,
        strongly_connected_components as nx_strongly_connected_components,
        subgraph_view as nx_subgraph_view,
        topological_sort as nx_topological_sort,
        tournament as nx_tournament,
        triadic_census as nx_triadic_census,
        voronoi_cells as nx_voronoi_cells,
        weakly_connected_components as nx_weakly_connected_components,
        wiener_index as nx_wiener_index,
    )

    numpy = imported_import_module("numpy")
    scipy = imported_import_module("scipy")
except ImportError:
    sys_stderr.write(
        f"NOTE: Please install networkx, numpy, scipy to be able to "
        f"use proxied NetworkX algorithms. E.g., CALL nxalg.pagerank(...).\n"
        f"Using Python:\n{sys_version}\n"
    )
    raise


# Optional nullable procedure arguments default to Cypher null, the only default Memgraph admits for a nullable
# argument (a vertex default is forbidden and a Boolean does not satisfy int, str or list). Null is also NetworkX's
# own "option absent" value, so an omitted option reaches NetworkX exactly as if the caller had not supplied it,
# while explicit zero depths/cutoffs and other concrete values are forwarded unchanged.


# networkx.algorithms.approximation.connectivity.node_connectivity
@mgp_read_proc
def node_connectivity(
    ctx: mgp_ProcCtx,
    source: mgp_Nullable[mgp_Vertex] = None,
    target: mgp_Nullable[mgp_Vertex] = None,
) -> mgp_Record(connectivity=int):
    computed_return_value = mgp_Record(connectivity=nx_node_connectivity(MemgraphMultiDiGraph(ctx=ctx), source, target))
    return computed_return_value


# networkx.algorithms.assortativity.degree_assortativity_coefficient
@mgp_read_proc
def degree_assortativity_coefficient(
    ctx: mgp_ProcCtx,
    x: str = "out",
    y: str = "in",
    weight: mgp_Nullable[str] = None,
    nodes: mgp_Nullable[mgp_List[mgp_Vertex]] = None,
) -> mgp_Record(assortativity=float):
    computed_return_value = mgp_Record(
        assortativity=nx_degree_assortativity_coefficient(MemgraphMultiDiGraph(ctx=ctx), x, y, weight, nodes)
    )
    return computed_return_value


# networkx.algorithms.asteroidal.is_at_free
@mgp_read_proc
def is_at_free(ctx: mgp_ProcCtx) -> mgp_Record(is_at_free=bool):
    computed_return_value = mgp_Record(is_at_free=nx_is_at_free(MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True)))
    return computed_return_value


# networkx.algorithms.bipartite.basic.is_bipartite
@mgp_read_proc
def is_bipartite(ctx: mgp_ProcCtx) -> mgp_Record(is_bipartite=bool):
    computed_return_value = mgp_Record(is_bipartite=nx_is_bipartite(MemgraphMultiDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.boundary.node_boundary
@mgp_read_proc
def node_boundary(
    ctx: mgp_ProcCtx,
    nbunch1: mgp_List[mgp_Vertex],
    nbunch2: mgp_Nullable[mgp_List[mgp_Vertex]] = None,
) -> mgp_Record(boundary=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(boundary=list(nx_node_boundary(MemgraphMultiDiGraph(ctx=ctx), nbunch1, nbunch2)))
    return computed_return_value


# networkx.algorithms.bridges.bridges
@mgp_read_proc
def bridges(ctx: mgp_ProcCtx, root: mgp_Nullable[mgp_Vertex] = None) -> mgp_Record(bridges=mgp_List[mgp_Edge]):
    g = MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True)
    simple_graph = MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True)
    computed_return_value = mgp_Record(bridges=[next(iter(g[u][v])) for u, v in nx_bridges(simple_graph, root=root)])
    return computed_return_value


# networkx.algorithms.centrality.betweenness_centrality
@mgp_read_proc
def betweenness_centrality(
    ctx: mgp_ProcCtx,
    k: mgp_Nullable[int] = None,
    normalized: bool = True,
    weight: mgp_Nullable[str] = None,
    endpoints: bool = False,
    seed: mgp_Nullable[int] = None,
) -> mgp_Record(node=mgp_Vertex, betweenness=mgp_Number):
    computed_return_value = [
        mgp_Record(node=n, betweenness=b)
        for n, b in nx_betweenness_centrality(
            MemgraphDiGraph(ctx=ctx),
            k=k,
            normalized=normalized,
            weight=weight,
            endpoints=endpoints,
            seed=seed,
        ).items()
    ]
    return computed_return_value


# networkx.algorithms.chains.chain_decomposition
@mgp_read_proc
def chain_decomposition(ctx: mgp_ProcCtx, root: mgp_Nullable[mgp_Vertex] = None) -> mgp_Record(chains=mgp_List[mgp_List[mgp_Edge]]):
    g = MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True)
    simple_graph = MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True)
    computed_return_value = mgp_Record(
        chains=[[next(iter(g[u][v])) for u, v in d] for d in nx_chain_decomposition(simple_graph, root=root)]
    )
    return computed_return_value


# networkx.algorithms.chordal.is_chordal
@mgp_read_proc
def is_chordal(ctx: mgp_ProcCtx) -> mgp_Record(is_chordal=bool):
    computed_return_value = mgp_Record(is_chordal=nx_is_chordal(MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True)))
    return computed_return_value


# networkx.algorithms.clique.find_cliques
@mgp_read_proc
def find_cliques(
    ctx: mgp_ProcCtx,
) -> mgp_Record(cliques=mgp_List[mgp_List[mgp_Vertex]]):
    computed_return_value = mgp_Record(cliques=list(nx_find_cliques(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True))))
    return computed_return_value


# networkx.algorithms.cluster.clustering
@mgp_read_proc
def clustering(
    ctx: mgp_ProcCtx,
    nodes: mgp_Nullable[mgp_List[mgp_Vertex]] = None,
    weight: mgp_Nullable[str] = None,
) -> mgp_Record(node=mgp_Vertex, clustering=mgp_Number):
    # A list or null `nodes` always makes NetworkX return a per-node dict.
    clustering_values = nx_clustering(MemgraphDiGraph(ctx=ctx), nodes=nodes, weight=weight)
    if not isinstance(clustering_values, dict):
        raise RuntimeError("NetworkX clustering returned one value where a per-node mapping was expected")
    computed_return_value = [mgp_Record(node=n, clustering=c) for n, c in clustering_values.items()]
    return computed_return_value


# networkx.algorithms.coloring.greedy_color
@mgp_read_proc
def greedy_color(
    ctx: mgp_ProcCtx,
    strategy: str = "largest_first",
    interchange: bool = False,
) -> mgp_Record(node=mgp_Vertex, color=int):
    computed_return_value = [
        mgp_Record(node=n, color=c) for n, c in nx_greedy_color(MemgraphMultiDiGraph(ctx=ctx), strategy, interchange).items()
    ]
    return computed_return_value


# networkx.algorithms.communicability_alg.communicability
@mgp_read_proc
def communicability(
    ctx: mgp_ProcCtx,
) -> mgp_Record(node1=mgp_Vertex, node2=mgp_Vertex, communicability=mgp_Number):
    computed_return_value = [
        mgp_Record(node1=n1, node2=n2, communicability=v)
        for n1, d in nx_communicability(MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True)).items()
        for n2, v in d.items()
    ]
    return computed_return_value


# networkx.algorithms.community.kclique.k_clique_communities
@mgp_read_proc
def k_clique_communities(
    ctx: mgp_ProcCtx,
    k: int,
    cliques: mgp_Nullable[mgp_List[mgp_List[mgp_Vertex]]] = None,
) -> mgp_Record(communities=mgp_List[mgp_List[mgp_Vertex]]):
    graph = MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True)
    computed_return_value = mgp_Record(communities=[list(s) for s in nx_community.k_clique_communities(graph, k, cliques)])
    return computed_return_value


# networkx.algorithms.approximation.kcomponents.k_components
@mgp_read_proc
def k_components(ctx: mgp_ProcCtx, density: mgp_Number = 0.95) -> mgp_Record(k=int, components=mgp_List[mgp_List[mgp_Vertex]]):
    kcomps = nx_k_components(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True), density)

    computed_return_value = [mgp_Record(k=k, components=[list(s) for s in comps]) for k, comps in kcomps.items()]
    return computed_return_value


# networkx.algorithms.components.biconnected_components
@mgp_read_proc
def biconnected_components(
    ctx: mgp_ProcCtx,
) -> mgp_Record(components=mgp_List[mgp_List[mgp_Vertex]]):
    comps = nx_biconnected_components(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True))
    computed_return_value = mgp_Record(components=[list(s) for s in comps])
    return computed_return_value


# networkx.algorithms.components.strongly_connected_components
@mgp_read_proc
def strongly_connected_components(
    ctx: mgp_ProcCtx,
) -> mgp_Record(components=mgp_List[mgp_List[mgp_Vertex]]):
    comps = nx_strongly_connected_components(MemgraphMultiDiGraph(ctx=ctx))
    computed_return_value = mgp_Record(components=[list(s) for s in comps])
    return computed_return_value


# networkx.algorithms.connectivity.edge_kcomponents.k_edge_components
#
# NOTE: NetworkX 2.4, algorithms/connectivity/edge_kcomponents.py:367. We create
# a *copy* of the graph because the algorithm copies the graph using
# __class__() and tries to modify it.
@mgp_read_proc
def k_edge_components(ctx: mgp_ProcCtx, k: int) -> mgp_Record(components=mgp_List[mgp_List[mgp_Vertex]]):
    computed_return_value = mgp_Record(components=[list(s) for s in nx_k_edge_components(nx_DiGraph(MemgraphDiGraph(ctx=ctx)), k)])
    return computed_return_value


# networkx.algorithms.core.core_number
@mgp_read_proc
def core_number(ctx: mgp_ProcCtx) -> mgp_Record(node=mgp_Vertex, core=mgp_Number):
    computed_return_value = [mgp_Record(node=n, core=c) for n, c in nx_core_number(MemgraphDiGraph(ctx=ctx)).items()]
    return computed_return_value


# networkx.algorithms.covering.is_edge_cover
@mgp_read_proc
def is_edge_cover(ctx: mgp_ProcCtx, cover: mgp_List[mgp_Edge]) -> mgp_Record(is_edge_cover=bool):
    cover = set([(e.from_vertex, e.to_vertex) for e in cover])
    computed_return_value = mgp_Record(
        is_edge_cover=nx_is_edge_cover(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True), cover)
    )
    return computed_return_value


# networkx.algorithms.cycles.find_cycle
@mgp_read_proc
def find_cycle(
    ctx: mgp_ProcCtx,
    source: mgp_Nullable[mgp_List[mgp_Vertex]] = None,
    orientation: mgp_Nullable[str] = None,
) -> mgp_Record(cycle=mgp_Nullable[mgp_List[mgp_Edge]]):
    try:
        computed_return_value = mgp_Record(
            cycle=[e for _, _, e in nx_find_cycle(MemgraphMultiDiGraph(ctx=ctx), source, orientation)]
        )
        return computed_return_value
    except nx_NetworkXNoCycle:
        computed_return_value = mgp_Record(cycle=None)
        return computed_return_value


# networkx.algorithms.cycles.simple_cycles
#
# NOTE: NetworkX 2.4, algorithms/cycles.py:183. We create a *copy* of the graph
# because the algorithm copies the graph using type() and tries to pass initial
# data.
@mgp_read_proc
def simple_cycles(
    ctx: mgp_ProcCtx,
) -> mgp_Record(cycles=mgp_List[mgp_List[mgp_Vertex]]):
    computed_return_value = mgp_Record(cycles=list(nx_simple_cycles(nx_MultiDiGraph(MemgraphMultiDiGraph(ctx=ctx)).copy())))
    return computed_return_value


# networkx.algorithms.cuts.node_expansion
@mgp_read_proc
def node_expansion(ctx: mgp_ProcCtx, s: mgp_List[mgp_Vertex]) -> mgp_Record(node_expansion=mgp_Number):
    computed_return_value = mgp_Record(node_expansion=nx_node_expansion(MemgraphMultiDiGraph(ctx=ctx), set(s)))
    return computed_return_value


# networkx.algorithms.dag.topological_sort
@mgp_read_proc
def topological_sort(
    ctx: mgp_ProcCtx,
) -> mgp_Record(nodes=mgp_Nullable[mgp_List[mgp_Vertex]]):
    computed_return_value = mgp_Record(nodes=list(nx_topological_sort(MemgraphMultiDiGraph(ctx=ctx))))
    return computed_return_value


# networkx.algorithms.dag.ancestors
@mgp_read_proc
def ancestors(ctx: mgp_ProcCtx, source: mgp_Vertex) -> mgp_Record(ancestors=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(ancestors=list(nx_ancestors(MemgraphMultiDiGraph(ctx=ctx), source)))
    return computed_return_value


# networkx.algorithms.dag.descendants
@mgp_read_proc
def descendants(ctx: mgp_ProcCtx, source: mgp_Vertex) -> mgp_Record(descendants=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(descendants=list(nx_descendants(MemgraphMultiDiGraph(ctx=ctx), source)))
    return computed_return_value


# networkx.algorithms.distance_measures.center
#
# NOTE: Takes more parameters.
@mgp_read_proc
def center(ctx: mgp_ProcCtx) -> mgp_Record(center=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(center=list(nx_center(MemgraphMultiDiGraph(ctx=ctx))))
    return computed_return_value


# networkx.algorithms.distance_measures.diameter
#
# NOTE: Takes more parameters.
@mgp_read_proc
def diameter(ctx: mgp_ProcCtx) -> mgp_Record(diameter=int):
    computed_return_value = mgp_Record(diameter=nx_diameter(MemgraphMultiDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.distance_regular.is_distance_regular
@mgp_read_proc
def is_distance_regular(ctx: mgp_ProcCtx) -> mgp_Record(is_distance_regular=bool):
    computed_return_value = mgp_Record(
        is_distance_regular=nx_is_distance_regular(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True))
    )
    return computed_return_value


# networkx.algorithms.strongly_regular.is_strongly_regular
@mgp_read_proc
def is_strongly_regular(ctx: mgp_ProcCtx) -> mgp_Record(is_strongly_regular=bool):
    computed_return_value = mgp_Record(
        is_strongly_regular=nx_is_strongly_regular(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True))
    )
    return computed_return_value


# networkx.algorithms.dominance.dominance_frontiers
@mgp_read_proc
def dominance_frontiers(
    ctx: mgp_ProcCtx,
    start: mgp_Vertex,
) -> mgp_Record(node=mgp_Vertex, frontier=mgp_List[mgp_Vertex]):
    computed_return_value = [
        mgp_Record(node=n, frontier=list(f)) for n, f in nx_dominance_frontiers(MemgraphMultiDiGraph(ctx=ctx), start).items()
    ]
    return computed_return_value


# networkx.algorithms.dominance.immediate_dominators
@mgp_read_proc
def immediate_dominators(
    ctx: mgp_ProcCtx,
    start: mgp_Vertex,
) -> mgp_Record(node=mgp_Vertex, dominator=mgp_Vertex):
    computed_return_value = [
        mgp_Record(node=n, dominator=d) for n, d in nx_immediate_dominators(MemgraphMultiDiGraph(ctx=ctx), start).items()
    ]
    return computed_return_value


# networkx.algorithms.dominating.dominating_set
@mgp_read_proc
def dominating_set(
    ctx: mgp_ProcCtx,
    start: mgp_Vertex,
) -> mgp_Record(dominating_set=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(dominating_set=list(nx_dominating_set(MemgraphMultiDiGraph(ctx=ctx), start)))
    return computed_return_value


# networkx.algorithms.efficiency_measures.local_efficiency
@mgp_read_proc
def local_efficiency(ctx: mgp_ProcCtx) -> mgp_Record(local_efficiency=float):
    computed_return_value = mgp_Record(
        local_efficiency=nx_local_efficiency(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True))
    )
    return computed_return_value


# networkx.algorithms.efficiency_measures.global_efficiency
@mgp_read_proc
def global_efficiency(ctx: mgp_ProcCtx) -> mgp_Record(global_efficiency=float):
    computed_return_value = mgp_Record(
        global_efficiency=nx_global_efficiency(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True))
    )
    return computed_return_value


# networkx.algorithms.euler.is_eulerian
@mgp_read_proc
def is_eulerian(ctx: mgp_ProcCtx) -> mgp_Record(is_eulerian=bool):
    computed_return_value = mgp_Record(is_eulerian=nx_is_eulerian(MemgraphMultiDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.euler.is_semieulerian
@mgp_read_proc
def is_semieulerian(ctx: mgp_ProcCtx) -> mgp_Record(is_semieulerian=bool):
    computed_return_value = mgp_Record(is_semieulerian=nx_is_semieulerian(MemgraphMultiDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.euler.has_eulerian_path
@mgp_read_proc
def has_eulerian_path(ctx: mgp_ProcCtx) -> mgp_Record(has_eulerian_path=bool):
    computed_return_value = mgp_Record(has_eulerian_path=nx_has_eulerian_path(MemgraphMultiDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.hierarchy.flow_hierarchy
@mgp_read_proc
def flow_hierarchy(ctx: mgp_ProcCtx, weight: mgp_Nullable[str] = None) -> mgp_Record(flow_hierarchy=float):
    computed_return_value = mgp_Record(flow_hierarchy=nx_flow_hierarchy(MemgraphMultiDiGraph(ctx=ctx), weight=weight))
    return computed_return_value


# networkx.algorithms.isolate.isolates
@mgp_read_proc
def isolates(ctx: mgp_ProcCtx) -> mgp_Record(isolates=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(isolates=list(nx_isolates(MemgraphMultiDiGraph(ctx=ctx))))
    return computed_return_value


# networkx.algorithms.isolate.is_isolate
@mgp_read_proc
def is_isolate(ctx: mgp_ProcCtx, n: mgp_Vertex) -> mgp_Record(is_isolate=bool):
    computed_return_value = mgp_Record(is_isolate=nx_is_isolate(MemgraphMultiDiGraph(ctx=ctx), n))
    return computed_return_value


# networkx.algorithms.isomorphism.is_isomorphic
@mgp_read_proc
def is_isomorphic(
    ctx: mgp_ProcCtx,
    nodes1: mgp_List[mgp_Vertex],
    edges1: mgp_List[mgp_Edge],
    nodes2: mgp_List[mgp_Vertex],
    edges2: mgp_List[mgp_Edge],
) -> mgp_Record(is_isomorphic=bool):
    nodes1, edges1, nodes2, edges2 = map(set, [nodes1, edges1, nodes2, edges2])
    g = MemgraphMultiDiGraph(ctx=ctx)
    g1 = nx_subgraph_view(g, filter_node=lambda n: n in nodes1, filter_edge=lambda n1, n2, e: e in edges1)
    g2 = nx_subgraph_view(g, filter_node=lambda n: n in nodes2, filter_edge=lambda n1, n2, e: e in edges2)
    computed_return_value = mgp_Record(is_isomorphic=nx_is_isomorphic(g1, g2))
    return computed_return_value


# networkx.algorithms.link_analysis.pagerank_alg.pagerank
@mgp_read_proc
def pagerank(
    ctx: mgp_ProcCtx,
    alpha: mgp_Number = 0.85,
    personalization: mgp_Nullable[str] = None,
    max_iter: int = 100,
    tol: mgp_Number = 1e-06,
    nstart: mgp_Nullable[str] = None,
    weight: mgp_Nullable[str] = "weight",
    dangling: mgp_Nullable[str] = None,
) -> mgp_Record(node=mgp_Vertex, rank=float):
    pagerank_arguments = dict(alpha=alpha, max_iter=max_iter, tol=tol, weight=weight)
    if isinstance(personalization, str) and personalization:
        pagerank_arguments["personalization"] = PropertiesDictionary(ctx, personalization)
    if isinstance(nstart, str) and nstart:
        pagerank_arguments["nstart"] = PropertiesDictionary(ctx, nstart)
    if isinstance(dangling, str) and dangling:
        pagerank_arguments["dangling"] = PropertiesDictionary(ctx, dangling)

    pg = nx_pagerank(MemgraphDiGraph(ctx=ctx), **pagerank_arguments)

    computed_return_value = [mgp_Record(node=k, rank=v) for k, v in pg.items()]
    return computed_return_value


# networkx.algorithms.link_prediction.jaccard_coefficient
@mgp_read_proc
def jaccard_coefficient(
    ctx: mgp_ProcCtx,
    ebunch: mgp_Nullable[mgp_List[mgp_List[mgp_Vertex]]] = None,
) -> mgp_Record(u=mgp_Vertex, v=mgp_Vertex, coef=float):
    graph = MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True)
    computed_return_value = [mgp_Record(u=u, v=v, coef=c) for u, v, c in nx_jaccard_coefficient(graph, ebunch)]
    return computed_return_value


# networkx.algorithms.lowest_common_ancestors.lowest_common_ancestor
@mgp_read_proc
def lowest_common_ancestor(ctx: mgp_ProcCtx, node1: mgp_Vertex, node2: mgp_Vertex) -> mgp_Record(ancestor=mgp_Nullable[mgp_Vertex]):
    computed_return_value = mgp_Record(ancestor=nx_lowest_common_ancestor(MemgraphDiGraph(ctx=ctx), node1, node2))
    return computed_return_value


# networkx.algorithms.matching.maximal_matching
@mgp_read_proc
def maximal_matching(ctx: mgp_ProcCtx) -> mgp_Record(edges=mgp_List[mgp_Edge]):
    # Matching is defined on the undirected simple graph: parallel edges and both directions of a vertex pair are one
    # candidate pair. Each matched pair is mapped back to one original host edge through the undirected multigraph
    # view, whose adjacency holds every host edge between the pair in either direction.
    matching = nx_maximal_matching(MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True))
    host_edges = MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True)
    computed_return_value = mgp_Record(edges=[next(iter(host_edges[u][v])) for u, v in matching])
    return computed_return_value


# networkx.algorithms.planarity.check_planarity
#
# NOTE: Returns a graph.
@mgp_read_proc
def check_planarity(ctx: mgp_ProcCtx) -> mgp_Record(is_planar=bool):
    computed_return_value = mgp_Record(is_planar=nx_check_planarity(MemgraphMultiDiGraph(ctx=ctx))[0])
    return computed_return_value


# networkx.algorithms.non_randomness.non_randomness
@mgp_read_proc
def non_randomness(
    ctx: mgp_ProcCtx,
    k: mgp_Nullable[int] = None,
) -> mgp_Record(non_randomness=float, relative_non_randomness=float):
    nn, rnn = nx_non_randomness(MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True), k=k)
    computed_return_value = mgp_Record(non_randomness=nn, relative_non_randomness=rnn)
    return computed_return_value


# networkx.algorithms.reciprocity.reciprocity
@mgp_read_proc
def reciprocity(
    ctx: mgp_ProcCtx,
    nodes: mgp_Nullable[mgp_List[mgp_Vertex]] = None,
) -> mgp_Record(node=mgp_Nullable[mgp_Vertex], reciprocity=mgp_Nullable[float]):
    rp = nx_reciprocity(MemgraphMultiDiGraph(ctx=ctx), nodes=nodes)
    if nodes is None:
        computed_return_value = [mgp_Record(node=None, reciprocity=rp)]
        return computed_return_value
    else:
        if not isinstance(rp, dict):
            raise TypeError("node reciprocity must return a dictionary")
        computed_return_value = [mgp_Record(node=n, reciprocity=r) for n, r in rp.items()]
        return computed_return_value


# networkx.algorithms.shortest_paths.generic.shortest_path
@mgp_read_proc
def shortest_path(
    ctx: mgp_ProcCtx,
    source: mgp_Nullable[mgp_Vertex] = None,
    target: mgp_Nullable[mgp_Vertex] = None,
    weight: mgp_Nullable[str] = None,
    method: str = "dijkstra",
) -> mgp_Record(source=mgp_Vertex, target=mgp_Vertex, path=mgp_List[mgp_Vertex]):
    sp = nx_shortest_path(
        MemgraphMultiDiGraph(ctx=ctx),
        source=source,
        target=target,
        weight=weight,
        method=method,
    )

    has_source = source is not None
    has_target = target is not None
    if has_source and has_target:
        sp = {source: {target: sp}}
    elif has_source:
        sp = {source: sp}
    elif has_target:
        sp = {s: {target: p} for s, p in sp.items()}
    else:
        sp = dict(sp)

    computed_return_value = [mgp_Record(source=s, target=t, path=p) for s, d in sp.items() for t, p in d.items()]
    return computed_return_value


# networkx.algorithms.shortest_paths.generic.shortest_path_length
@mgp_read_proc
def shortest_path_length(
    ctx: mgp_ProcCtx,
    source: mgp_Nullable[mgp_Vertex] = None,
    target: mgp_Nullable[mgp_Vertex] = None,
    weight: mgp_Nullable[str] = None,
    method: str = "dijkstra",
) -> mgp_Record(source=mgp_Vertex, target=mgp_Vertex, length=mgp_Number):
    sp = nx_shortest_path_length(
        MemgraphMultiDiGraph(ctx=ctx),
        source=source,
        target=target,
        weight=weight,
        method=method,
    )

    has_source = source is not None
    has_target = target is not None
    if has_source and has_target:
        sp = {source: {target: sp}}
    elif has_source:
        sp = {source: sp}
    elif has_target:
        sp = {s: {target: local_element} for s, local_element in sp.items()}
    else:
        sp = dict(sp)

    computed_return_value = [
        mgp_Record(source=s, target=t, length=local_element) for s, d in sp.items() for t, local_element in d.items()
    ]
    return computed_return_value


# networkx.algorithms.shortest_paths.generic.all_shortest_paths
@mgp_read_proc
def all_shortest_paths(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    weight: mgp_Nullable[str] = None,
    method: str = "dijkstra",
) -> mgp_Record(paths=mgp_List[mgp_List[mgp_Vertex]]):
    computed_return_value = mgp_Record(
        paths=list(
            nx_all_shortest_paths(
                MemgraphMultiDiGraph(ctx=ctx),
                source=source,
                target=target,
                weight=weight,
                method=method,
            )
        )
    )
    return computed_return_value


# networkx.algorithms.shortest_paths.generic.has_path
@mgp_read_proc
def has_path(ctx: mgp_ProcCtx, source: mgp_Vertex, target: mgp_Vertex) -> mgp_Record(has_path=bool):
    computed_return_value = mgp_Record(has_path=nx_has_path(MemgraphMultiDiGraph(ctx=ctx), source, target))
    return computed_return_value


# networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra_path
@mgp_read_proc
def multi_source_dijkstra_path(
    ctx: mgp_ProcCtx,
    sources: mgp_List[mgp_Vertex],
    cutoff: mgp_Nullable[int] = None,
    weight: str = "weight",
) -> mgp_Record(target=mgp_Vertex, path=mgp_List[mgp_Vertex]):
    computed_return_value = [
        mgp_Record(target=t, path=p)
        for t, p in nx_multi_source_dijkstra_path(MemgraphMultiDiGraph(ctx=ctx), sources, cutoff=cutoff, weight=weight).items()
    ]
    return computed_return_value


# networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra_path_length
@mgp_read_proc
def multi_source_dijkstra_path_length(
    ctx: mgp_ProcCtx,
    sources: mgp_List[mgp_Vertex],
    cutoff: mgp_Nullable[int] = None,
    weight: str = "weight",
) -> mgp_Record(target=mgp_Vertex, length=mgp_Number):
    computed_return_value = [
        mgp_Record(target=t, length=local_element)
        for t, local_element in nx_multi_source_dijkstra_path_length(
            MemgraphMultiDiGraph(ctx=ctx), sources, cutoff=cutoff, weight=weight
        ).items()
    ]
    return computed_return_value


# networkx.algorithms.simple_paths.is_simple_path
@mgp_read_proc
def is_simple_path(ctx: mgp_ProcCtx, nodes: mgp_List[mgp_Vertex]) -> mgp_Record(is_simple_path=bool):
    computed_return_value = mgp_Record(is_simple_path=nx_is_simple_path(MemgraphMultiDiGraph(ctx=ctx), nodes))
    return computed_return_value


# networkx.algorithms.simple_paths.all_simple_paths
@mgp_read_proc
def all_simple_paths(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    target: mgp_Vertex,
    cutoff: mgp_Nullable[int] = None,
) -> mgp_Record(paths=mgp_List[mgp_List[mgp_Vertex]]):
    computed_return_value = mgp_Record(
        paths=list(nx_all_simple_paths(MemgraphMultiDiGraph(ctx=ctx), source, target, cutoff=cutoff))
    )
    return computed_return_value


# networkx.algorithms.tournament.is_tournament
@mgp_read_proc
def is_tournament(ctx: mgp_ProcCtx) -> mgp_Record(is_tournament=bool):
    computed_return_value = mgp_Record(is_tournament=nx_tournament.is_tournament(MemgraphDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.traversal.breadth_first_search.bfs_edges
@mgp_read_proc
def bfs_edges(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    reverse: bool = False,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(edges=mgp_List[mgp_Edge]):
    computed_return_value = mgp_Record(
        edges=list(
            nx_bfs_edges(
                MemgraphMultiDiGraph(ctx=ctx),
                source,
                reverse=reverse,
                depth_limit=depth_limit,
            )
        )
    )
    return computed_return_value


# networkx.algorithms.traversal.breadth_first_search.bfs_tree
@mgp_read_proc
def bfs_tree(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    reverse: bool = False,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(tree=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(
        tree=list(
            nx_bfs_tree(
                MemgraphMultiDiGraph(ctx=ctx),
                source,
                reverse=reverse,
                depth_limit=depth_limit,
            )
        )
    )
    return computed_return_value


# networkx.algorithms.traversal.breadth_first_search.bfs_predecessors
@mgp_read_proc
def bfs_predecessors(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(node=mgp_Vertex, predecessor=mgp_Vertex):
    computed_return_value = [
        mgp_Record(node=n, predecessor=p)
        for n, p in nx_bfs_predecessors(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit)
    ]
    return computed_return_value


# networkx.algorithms.traversal.breadth_first_search.bfs_successors
@mgp_read_proc
def bfs_successors(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(node=mgp_Vertex, successors=mgp_List[mgp_Vertex]):
    computed_return_value = [
        mgp_Record(node=n, successors=s)
        for n, s in nx_bfs_successors(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit)
    ]
    return computed_return_value


# networkx.algorithms.traversal.depth_first_search.dfs_tree
@mgp_read_proc
def dfs_tree(ctx: mgp_ProcCtx, source: mgp_Vertex, depth_limit: mgp_Nullable[int] = None) -> mgp_Record(tree=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(tree=list(nx_dfs_tree(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit)))
    return computed_return_value


# networkx.algorithms.traversal.depth_first_search.dfs_predecessors
@mgp_read_proc
def dfs_predecessors(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(node=mgp_Vertex, predecessor=mgp_Vertex):
    computed_return_value = [
        mgp_Record(node=n, predecessor=p)
        for n, p in nx_dfs_predecessors(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit).items()
    ]
    return computed_return_value


# networkx.algorithms.traversal.depth_first_search.dfs_successors
@mgp_read_proc
def dfs_successors(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(node=mgp_Vertex, successors=mgp_List[mgp_Vertex]):
    computed_return_value = [
        mgp_Record(node=n, successors=s)
        for n, s in nx_dfs_successors(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit).items()
    ]
    return computed_return_value


# networkx.algorithms.traversal.depth_first_search.dfs_preorder_nodes
@mgp_read_proc
def dfs_preorder_nodes(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(nodes=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(
        nodes=list(nx_dfs_preorder_nodes(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit))
    )
    return computed_return_value


# networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes
@mgp_read_proc
def dfs_postorder_nodes(
    ctx: mgp_ProcCtx,
    source: mgp_Vertex,
    depth_limit: mgp_Nullable[int] = None,
) -> mgp_Record(nodes=mgp_List[mgp_Vertex]):
    computed_return_value = mgp_Record(
        nodes=list(nx_dfs_postorder_nodes(MemgraphMultiDiGraph(ctx=ctx), source, depth_limit=depth_limit))
    )
    return computed_return_value


# networkx.algorithms.traversal.edgebfs.edge_bfs
@mgp_read_proc
def edge_bfs(
    ctx: mgp_ProcCtx,
    source: mgp_Nullable[mgp_Vertex] = None,
    orientation: mgp_Nullable[str] = None,
) -> mgp_Record(edges=mgp_List[mgp_Edge]):
    computed_return_value = mgp_Record(
        edges=list(e for _, _, e in nx_edge_bfs(MemgraphMultiDiGraph(ctx=ctx), source=source, orientation=orientation))
    )
    return computed_return_value


# networkx.algorithms.traversal.edgedfs.edge_dfs
@mgp_read_proc
def edge_dfs(
    ctx: mgp_ProcCtx,
    source: mgp_Nullable[mgp_Vertex] = None,
    orientation: mgp_Nullable[str] = None,
) -> mgp_Record(edges=mgp_List[mgp_Edge]):
    computed_return_value = mgp_Record(
        edges=list(e for _, _, e in nx_edge_dfs(MemgraphMultiDiGraph(ctx=ctx), source=source, orientation=orientation))
    )
    return computed_return_value


# networkx.algorithms.tree.recognition.is_tree
@mgp_read_proc
def is_tree(ctx: mgp_ProcCtx) -> mgp_Record(is_tree=bool):
    computed_return_value = mgp_Record(is_tree=nx_is_tree(MemgraphDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.tree.recognition.is_forest
@mgp_read_proc
def is_forest(ctx: mgp_ProcCtx) -> mgp_Record(is_forest=bool):
    computed_return_value = mgp_Record(is_forest=nx_is_forest(MemgraphDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.tree.recognition.is_arborescence
@mgp_read_proc
def is_arborescence(ctx: mgp_ProcCtx) -> mgp_Record(is_arborescence=bool):
    computed_return_value = mgp_Record(is_arborescence=nx_is_arborescence(MemgraphDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.tree.recognition.is_branching
@mgp_read_proc
def is_branching(ctx: mgp_ProcCtx) -> mgp_Record(is_branching=bool):
    computed_return_value = mgp_Record(is_branching=nx_is_branching(MemgraphDiGraph(ctx=ctx)))
    return computed_return_value


# networkx.algorithms.tree.mst.minimum_spanning_tree
@mgp_read_proc
def minimum_spanning_tree(
    ctx: mgp_ProcCtx,
    weight: str = "weight",
    algorithm: str = "kruskal",
    ignore_nan: bool = False,
) -> mgp_Record(nodes=mgp_List[mgp_Vertex], edges=mgp_List[mgp_Edge]):
    gres = nx_minimum_spanning_tree(MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True), weight, algorithm, ignore_nan)
    computed_return_value = mgp_Record(nodes=list(gres.nodes()), edges=[e for _, _, e in gres.edges(keys=True)])
    return computed_return_value


# networkx.algorithms.triads.triadic_census
@mgp_read_proc
def triadic_census(ctx: mgp_ProcCtx) -> mgp_Record(triad=str, count=int):
    computed_return_value = [mgp_Record(triad=t, count=c) for t, c in nx_triadic_census(MemgraphDiGraph(ctx=ctx)).items()]
    return computed_return_value


# networkx.algorithms.voronoi.voronoi_cells
@mgp_read_proc
def voronoi_cells(
    ctx: mgp_ProcCtx,
    center_nodes: mgp_List[mgp_Vertex],
    weight: str = "weight",
) -> mgp_Record(center=mgp_Vertex, cell=mgp_List[mgp_Vertex]):
    computed_return_value = [
        mgp_Record(center=c1, cell=list(c2))
        for c1, c2 in nx_voronoi_cells(MemgraphMultiDiGraph(ctx=ctx), center_nodes, weight=weight).items()
    ]
    return computed_return_value


# networkx.algorithms.wiener.wiener_index
@mgp_read_proc
def wiener_index(ctx: mgp_ProcCtx, weight: mgp_Nullable[str] = None) -> mgp_Record(wiener_index=mgp_Number):
    computed_return_value = mgp_Record(wiener_index=nx_wiener_index(MemgraphMultiDiGraph(ctx=ctx), weight=weight))
    return computed_return_value


@mgp_read_proc
def weakly_connected_components_subgraph(
    vertices: mgp_List[mgp_Vertex],
    edges: mgp_List[mgp_Edge],
) -> mgp_Record(n_components=int, components=mgp_List[mgp_List[mgp_Vertex]]):
    """
    This procedure finds weakly connected components of a given subgraph of a
    directed graph.

    The subgraph is defined by a list of vertices and a list edges which are
    passed as arguments of the procedure. More precisely, a set of vertices of
    a subgraph contains all vertices provided in a list of vertices along with
    all vertices that are endpoints of provided edges. Similarly, a set of
    edges of a subgraph contains all edges from the list of provided edges.

    The procedure returns 2 fields:
        * `n_components` is the number of weakly connected components of the
        subgraph.
        * `components` is a list of weakly connected components. Each component
        is given as a list of `mgp.Vertex` objects from that component.

    For example, weakly connected components in a subgraph formed from all
    vertices labeled `Person` and edges between such vertices can be obtained
    using the following openCypher query:

    MATCH (n:Person)-[e]->(m:Person)
    WITH collect(n) AS nodes, collect(e) AS edges
    CALL wcc.get_components(nodes, edges) YIELD *
    RETURN n_components, components;
    """
    g = nx_DiGraph()
    g.add_nodes_from(vertices)
    g.add_edges_from([(edge.from_vertex, edge.to_vertex) for edge in edges])

    components = [list(wcc) for wcc in nx_weakly_connected_components(g)]

    computed_return_value = mgp_Record(n_components=len(components), components=components)
    return computed_return_value
