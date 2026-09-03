"""Utilities for mgp igraph."""

from collections import defaultdict
from enum import Enum as enum_Enum

from igraph import EdgeSeq as igraph_EdgeSeq
from igraph import Graph as igraph_Graph


class MemgraphIgraph(igraph_Graph):
    def __init__(self, ctx, directed: bool):
        if not hasattr(ctx, "graph"):
            raise TypeError("MemgraphIgraph requires a context with a graph")
        self.ctx_graph = ctx.graph
        self.id_mappings, self.reverse_id_mappings = self.create_igraph_from_ctx(ctx=ctx, directed=directed)

    def get_igraph_vertex_id(self, vertex) -> int:
        if not hasattr(vertex, "id"):
            raise TypeError("Memgraph vertices must provide an integer id")
        vertex_id = vertex.id
        if not isinstance(vertex_id, int):
            raise TypeError(f"Memgraph vertex id must be an int, received {type(vertex_id)}")
        igraph_vertex_id = self.id_mappings.get(vertex_id, -1)
        if igraph_vertex_id < 0:
            raise KeyError(f"Memgraph vertex {vertex_id} is absent from the igraph mapping")
        return igraph_vertex_id

    def maxflow(self, source, target, capacity: str) -> float:
        flow = super().maxflow(
            self.get_igraph_vertex_id(source),
            self.get_igraph_vertex_id(target),
            capacity=capacity,
        )
        return flow.value

    def pagerank(
        self,
        weights: str,
        directed: bool,
        damping: float,
        implementation: str,
    ) -> list[tuple[object, float]]:
        pagerank_values = super().pagerank(
            weights=weights,
            directed=directed,
            damping=damping,
            implementation=implementation,
        )

        computed_return_value = [(self.get_vertex_by_id(node_id), rank) for node_id, rank in enumerate(pagerank_values)]
        return computed_return_value

    def get_all_simple_paths(self, v, to, cutoff: int) -> list[list[object]]:
        if cutoff < 0:
            raise ValueError(f"Path cutoff must be non-negative, received {cutoff}")
        paths = [
            self.convert_vertex_ids_to_mgp_vertices(path)
            for path in super().get_all_simple_paths(
                v=self.get_igraph_vertex_id(v),
                to=self.get_igraph_vertex_id(to),
                maxlen=cutoff,
            )
        ]
        return paths

    def topological_sort(self, mode: str) -> list[object]:
        sorted_vertex_ids = super().topological_sorting(mode=mode)
        computed_return_value = self.convert_vertex_ids_to_mgp_vertices(sorted_vertex_ids)
        return computed_return_value

    def community_leiden(
        self,
        resolution_parameter,
        weights,
        n_iterations,
        beta=0.01,
        objective_function="CPM",
        initial_membership=False,
        node_weights=False,
    ) -> list[tuple[object, int]]:
        parameters = {
            "resolution": resolution_parameter,
            "n_iterations": n_iterations,
            "objective_function": objective_function,
            "beta": beta,
        }
        if weights:
            parameters["weights"] = weights
        if initial_membership:
            parameters["initial_membership"] = initial_membership
        if node_weights:
            parameters["node_weights"] = node_weights
        communities = super().community_leiden(**parameters)
        computed_return_value = [(self.get_vertex_by_id(member), i) for i, members in enumerate(communities) for member in members]
        return computed_return_value

    def mincut(self, source, target, capacity: str) -> tuple[list[list[object]], float]:
        cut = super().mincut(
            source=self.get_igraph_vertex_id(source),
            target=self.get_igraph_vertex_id(target),
            capacity=capacity,
        )

        partition_vertices = [self.convert_vertex_ids_to_mgp_vertices(vertex_ids=partition) for partition in cut.partition]
        return partition_vertices, cut.value

    def spanning_tree(
        self,
        weights: str,
    ) -> list[list[object]]:
        if weights:
            min_spanning_tree_edges = super().spanning_tree(weights=self.es[weights], return_tree=False)
        else:
            min_spanning_tree_edges = super().spanning_tree(return_tree=False)

        computed_return_value = self.get_min_span_tree_vertex_pairs(min_spanning_tree_edges=self.es[min_spanning_tree_edges])
        return computed_return_value

    def shortest_path_length(self, source, target, weights: str) -> float:
        source_id = self.get_igraph_vertex_id(source)
        target_id = self.get_igraph_vertex_id(target)
        if weights:
            distances = super().distances(source=source_id, target=target_id, weights=weights)
        else:
            distances = super().distances(source=source_id, target=target_id)
        if not distances or not distances[0]:
            raise RuntimeError("igraph returned no shortest-path distance")
        length = distances[0][0]

        computed_return_value = float(length)
        return computed_return_value

    def all_shortest_path_lengths(self, weights: str) -> list[list[float]]:
        if weights:
            computed_return_value = super().distances(weights=weights)
        else:
            computed_return_value = super().distances()
        return computed_return_value

    def get_shortest_path(self, source, target, weights: str) -> list[object]:
        source_id = self.get_igraph_vertex_id(source)
        target_id = self.get_igraph_vertex_id(target)
        if weights:
            paths = super().get_shortest_paths(v=source_id, to=target_id, weights=weights)
        else:
            paths = super().get_shortest_paths(v=source_id, to=target_id)
        if not paths:
            raise RuntimeError("igraph returned no shortest path result")
        path = paths[0]

        computed_return_value = self.convert_vertex_ids_to_mgp_vertices(path)
        return computed_return_value

    def get_vertex_by_id(self, vertex_id: int) -> object:
        memgraph_vertex_id = self.reverse_id_mappings.get(vertex_id, -1)
        if memgraph_vertex_id < 0:
            raise KeyError(f"igraph vertex {vertex_id} is absent from the Memgraph mapping")
        computed_return_value = self.ctx_graph.get_vertex_by_id(memgraph_vertex_id)
        return computed_return_value

    def convert_vertex_ids_to_mgp_vertices(self, vertex_ids: list[int]) -> list[object]:
        vertices = []
        for vertex_id in vertex_ids:
            vertices.append(self.get_vertex_by_id(vertex_id))

        return vertices

    def get_min_span_tree_vertex_pairs(
        self,
        min_spanning_tree_edges: igraph_EdgeSeq,
    ) -> list[list[object]]:
        """Function for getting vertex pairs that are connected in minimum spanning tree.

        Args:
            min_span_tree_graph (igraph.EdgeSeq): Igraph graph containing minimum spanning tree

        Returns:
            List[List[mgp.Vertex]]: List of vertex pairs that are connected in minimum spanning tree
        """

        min_span_tree = []
        for edge in min_spanning_tree_edges:
            min_span_tree.append(
                [
                    self.get_vertex_by_id(edge.source),
                    self.get_vertex_by_id(edge.target),
                ]
            )

        return min_span_tree

    def create_igraph_from_ctx(self, ctx, directed: bool = False) -> tuple[dict[int, int], dict[int, int]]:
        (
            "Function for creating igraph.Graph from mgp.ProcCtx.\n\n        Args:\n            ctx (mgp.Pro"  # Continue literal.
            "cCtx): memgraph ProcCtx object\n            directed (bool, optional): Is graph directed. Def"  # Continue literal.
            "aults to False.\n\n        Returns:\n            Tuple[igraph.Graph, Dict[int, int], Dict[int, "  # Continue literal.
            "int]]: Returns Igraph.Graph object, vertex id mappings and inverted_id_mapping vertex id map"  # Continue literal.
            "pings\n"
        )

        vertex_attrs = defaultdict(list)
        edge_list = []
        edge_attrs = defaultdict(list)
        id_mapping = {vertex.id: i for i, vertex in enumerate(ctx.graph.vertices)}
        inverted_id_mapping = {i: vertex.id for i, vertex in enumerate(ctx.graph.vertices)}
        for vertex in ctx.graph.vertices:
            for name, value in vertex.properties.items():
                attribute_values = vertex_attrs.get(name, [])
                attribute_values.append(value)
                vertex_attrs[name] = attribute_values
            for edge in vertex.out_edges:
                for name, value in edge.properties.items():
                    attribute_values = edge_attrs.get(name, [])
                    attribute_values.append(value)
                    edge_attrs[name] = attribute_values
                source_id = id_mapping.get(edge.from_vertex.id, -1)
                target_id = id_mapping.get(edge.to_vertex.id, -1)
                if source_id < 0 or target_id < 0:
                    raise KeyError("Memgraph edge references a vertex outside the graph context")
                edge_list.append(
                    (
                        source_id,
                        target_id,
                    )
                )

        super().__init__(
            directed=directed,
            n=len(ctx.graph.vertices),
            edges=edge_list,
            edge_attrs=edge_attrs,
            vertex_attrs=vertex_attrs,
        )

        return id_mapping, inverted_id_mapping


class PageRankImplementationOptions(enum_Enum):
    PRPACK = "prpack"
    ARPACK = "arpack"


class InvalidPageRankImplementationOption(Exception):
    pass


class TopologicalSortingModes(enum_Enum):
    IN = "in"
    OUT = "out"


class InvalidTopologicalSortingModeException(Exception):
    pass


class CommunityDetectionObjectiveFunctionOptions(enum_Enum):
    CPM = "CPM"
    MODULARITY = "modularity"


class InvalidCommunityDetectionObjectiveFunctionException(Exception):
    pass


class TopologicalSortException(Exception):
    pass
