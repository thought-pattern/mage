"""Utilities for mgp networkx."""

from collections import abc as collections_abc
from sys import stderr as sys_stderr, version as sys_version

from mgp import Edge as mgp_Edge, Vertex as mgp_Vertex

try:
    from networkx import DiGraph as nx_DiGraph, MultiDiGraph as nx_MultiDiGraph
except ImportError:
    sys_stderr.write(f"NOTE: Please install networkx to be able touse graph_analyzer module. Using Python: {sys_version}")
    raise


class MemgraphAdjlistOuterDict(collections_abc.Mapping):
    __slots__ = ("internal_ctx", "internal_multi", "internal_succ")

    def __init__(self, ctx, succ=True, multi=True):
        self.internal_ctx = ctx
        self.internal_succ = succ
        self.internal_multi = multi

    def __getitem__(self, key):
        if key not in self:
            raise KeyError
        computed_return_value = MemgraphAdjlistInnerDict(key, succ=self.internal_succ, multi=self.internal_multi)
        return computed_return_value

    def __iter__(self):
        computed_return_value = iter(self.internal_ctx.graph.vertices)
        return computed_return_value

    def __len__(self):
        computed_return_value = len(self.internal_ctx.graph.vertices)
        return computed_return_value

    def __contains__(self, key):
        if not isinstance(key, mgp_Vertex):
            raise TypeError
        computed_return_value = key in self.internal_ctx.graph.vertices
        return computed_return_value


class MemgraphAdjlistInnerDict(collections_abc.Mapping):
    __slots__ = ("internal_multi", "internal_neighbors", "internal_node", "internal_succ")

    def __init__(self, node, succ=True, multi=True):
        self.internal_node = node
        self.internal_succ = succ
        self.internal_multi = multi
        self.internal_neighbors = []

    def __getitem__(self, key):
        # NOTE: NetworkX 2.4, classes/coreviews.py:143. UnionAtlas expects a
        # KeyError when indexing with a vertex that is not a neighbor.
        if key not in self:
            raise KeyError
        if not self.internal_multi:
            computed_return_value = UnhashableProperties(self.get_edge(key).properties)
            return computed_return_value
        computed_return_value = MemgraphEdgeKeyDict(self.internal_node, key, self.internal_succ)
        return computed_return_value

    def __iter__(self):
        yield from self.get_neighbors()

    def __len__(self):
        computed_return_value = len(self.get_neighbors())
        return computed_return_value

    def __contains__(self, key):
        if not isinstance(key, mgp_Vertex):
            raise TypeError
        computed_return_value = key in self.get_neighbors()
        return computed_return_value

    def get_neighbors(self):
        if not self.internal_neighbors:
            if self.internal_succ:
                self.internal_neighbors = set(e.to_vertex for e in self.internal_node.out_edges)
            else:
                self.internal_neighbors = set(e.from_vertex for e in self.internal_node.in_edges)
        return self.internal_neighbors

    def get_edge(self, neighbor):
        if self.internal_succ:
            edge = list(filter(lambda e: e.to_vertex == neighbor, self.internal_node.out_edges))
        else:
            edge = list(filter(lambda e: e.from_vertex == neighbor, self.internal_node.in_edges))

        assert len(edge) >= 1
        if len(edge) > 1:
            raise RuntimeError("Graph contains multiedges but is of non-multigraph type: {}".format(edge))

        computed_return_value = edge[0]
        return computed_return_value


class MemgraphEdgeKeyDict(collections_abc.Mapping):
    __slots__ = ("internal_edges", "internal_neighbor", "internal_node", "internal_succ")

    def __init__(self, node, neighbor, succ=True):
        self.internal_node = node
        self.internal_neighbor = neighbor
        self.internal_succ = succ
        self.internal_edges = []

    def __getitem__(self, key):
        if key not in self:
            raise KeyError
        computed_return_value = UnhashableProperties(key.properties)
        return computed_return_value

    def __iter__(self):
        yield from self.get_edges()

    def __len__(self):
        computed_return_value = len(self.get_edges())
        return computed_return_value

    def __contains__(self, key):
        if not isinstance(key, mgp_Edge):
            raise TypeError
        computed_return_value = key in self.get_edges()
        return computed_return_value

    def get_edges(self):
        if not self.internal_edges:
            if self.internal_succ:
                self.internal_edges = list(filter(lambda e: e.to_vertex == self.internal_neighbor, self.internal_node.out_edges))
            else:
                self.internal_edges = list(filter(lambda e: e.from_vertex == self.internal_neighbor, self.internal_node.in_edges))
        return self.internal_edges


class UnhashableProperties(collections_abc.Mapping):
    __slots__ = ("internal_properties",)

    def __init__(self, properties):
        self.internal_properties = properties

    def __getitem__(self, key):
        # Mapping protocol: a missing property raises KeyError, so callers'
        # explicit defaults such as NetworkX's get("weight", 1) apply.
        if key not in self.internal_properties:
            raise KeyError(key)
        computed_return_value = self.internal_properties.get(key, "")
        return computed_return_value

    def __iter__(self):
        yield from self.internal_properties

    def __len__(self):
        computed_return_value = len(self.internal_properties)
        return computed_return_value

    def __contains__(self, key):
        computed_return_value = key in self.internal_properties
        return computed_return_value

    def __hash__(self):
        raise TypeError(f"{type(self).__name__} is unhashable")


class MemgraphNodeDict(collections_abc.Mapping):
    __slots__ = ("internal_ctx",)

    def __init__(self, ctx):
        self.internal_ctx = ctx

    def __getitem__(self, key):
        if key not in self:
            raise KeyError
        # NOTE: NetworkX 2.4, classes/digraph.py:484. NetworkX expects the
        # tuples provided to add_nodes_from to be unhashable and cause a
        # TypeError when trying to index the node dictionary. This happens
        # because the data dictionary element of the tuple is unhashable. We do
        # the same thing by returning an unhashable data dictionary.
        computed_return_value = UnhashableProperties(key.properties)
        return computed_return_value

    def __iter__(self):
        computed_return_value = iter(self.internal_ctx.graph.vertices)
        return computed_return_value

    def __len__(self):
        computed_return_value = len(self.internal_ctx.graph.vertices)
        return computed_return_value

    def __contains__(self, key):
        # NOTE: NetworkX 2.4, graph.py:425. Graph.__contains__ relies on
        # self._node's (i.e. the dictionary produced by node_dict_factory)
        # __contains__ to raise a TypeError when indexing with something weird,
        # e.g. with sets. This is the behavior of dict.
        if not isinstance(key, mgp_Vertex):
            raise TypeError
        computed_return_value = key in self.internal_ctx.graph.vertices
        return computed_return_value


class MemgraphDiGraphBase:
    def __init__(self, incoming_graph_data=False, ctx=False, multi=True, **kwargs):
        # NOTE: We assume that our graph will never be given any initial data
        # because we already pull our data from the Memgraph database. This
        # assert is triggered by certain NetworkX procedures because they
        # create a new instance of our class and try to populate it with their
        # own data.
        if incoming_graph_data is None:
            incoming_graph_data = False
        assert incoming_graph_data is False

        # NOTE: We allow for ctx to be None in order to allow certain NetworkX
        # procedures (such as `subgraph`) to work. Such procedures directly
        # modify the graph's internal attributes and don't try to populate it
        # with initial data or modify it.

        # The *_factory attributes are NetworkX's documented subclass hooks, which its constructor calls to obtain the
        # graph's mappings; their names and zero-argument calling convention belong to the library.
        self.node_dict_factory = lambda: MemgraphNodeDict(ctx) if ctx else self.internal_error
        self.node_attr_dict_factory = self.internal_error

        self.adjlist_outer_dict_factory = lambda: MemgraphAdjlistOuterDict(ctx, multi=multi) if ctx else self.internal_error
        self.adjlist_inner_dict_factory = self.internal_error
        self.edge_key_dict_factory = self.internal_error
        self.edge_attr_dict_factory = self.internal_error

        # NOTE: We forbid any mutating operations because our graph is
        # immutable and pulls its data from the Memgraph database.
        for f in [
            "add_node",
            "add_nodes_from",
            "remove_node",
            "remove_nodes_from",
            "add_edge",
            "add_edges_from",
            "add_weighted_edges_from",
            "new_edge_key",
            "remove_edge",
            "remove_edges_from",
            "update",
            "clear",
        ]:
            setattr(self, f, lambda *args, **kwargs: self.internal_error())

        # NetworkX converts any incoming data other than its None default, so
        # the absent data is omitted rather than passed as a false value.
        super().__init__(**kwargs)

        # NOTE: This is a necessary hack because NetworkX assumes that the
        # customizable factory functions will only ever return *empty*
        # dictionaries. In our case, the factory functions return our custom,
        # already populated, dictionaries. Because self._pred and self._succ are
        # initialized by the same factory function, they end up storing the
        # same adjacency lists which is not good. We correct that here.
        # `_pred` is NetworkX's own predecessor attribute name: its algorithms read G._pred directly and the subclass
        # hooks have no separate predecessor factory, so the name is the library's protocol and is kept as is.
        self._pred = MemgraphAdjlistOuterDict(ctx, succ=False, multi=multi)

    def internal_error(self):
        raise RuntimeError("Modification operations are not supported")


class MemgraphMultiDiGraph(MemgraphDiGraphBase, nx_MultiDiGraph):
    """Read-only multigraph view of the host graph. Undirected algorithms use NetworkX's own undirected view of it,
    MemgraphMultiDiGraph(ctx=ctx).to_undirected(as_view=True), built at the call site."""

    def __init__(self, incoming_graph_data=False, ctx=False, **kwargs):
        super().__init__(incoming_graph_data=incoming_graph_data, ctx=ctx, multi=True, **kwargs)


class MemgraphDiGraph(MemgraphDiGraphBase, nx_DiGraph):
    """Read-only simple-graph view of the host graph. Undirected algorithms use NetworkX's own undirected view of it,
    MemgraphDiGraph(ctx=ctx).to_undirected(as_view=True), built at the call site."""

    def __init__(self, incoming_graph_data=False, ctx=False, **kwargs):
        super().__init__(incoming_graph_data=incoming_graph_data, ctx=ctx, multi=False, **kwargs)


class PropertiesDictionary(collections_abc.Mapping):
    __slots__ = ("internal_ctx", "internal_len", "internal_prop")

    def __init__(self, ctx, prop):
        self.internal_ctx = ctx
        self.internal_prop = prop
        # Cached property-bearing vertex count; zero means not yet counted (recounting an empty result is harmless).
        self.internal_len = 0

    def __getitem__(self, vertex):
        if vertex not in self:
            raise KeyError(f"{vertex} doesn't have the required property '{self.internal_prop}'")
        computed_return_value = vertex.properties.get(self.internal_prop, "")
        return computed_return_value

    def __iter__(self):
        for v in self.internal_ctx.graph.vertices:
            if self.internal_prop in v.properties:
                yield v

    def __len__(self):
        if not self.internal_len:
            self.internal_len = sum(1 for _ in self)
        return self.internal_len

    def __contains__(self, vertex):
        if not isinstance(vertex, mgp_Vertex):
            raise TypeError
        computed_return_value = self.internal_prop in vertex.properties
        return computed_return_value
