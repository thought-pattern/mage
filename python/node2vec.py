"""Utilities for node2vec."""

from inspect import cleandoc
from itertools import chain, repeat
from typing import Dict, List

from gensim import models as gensim_models
from mgp import List as mgp_List
from mgp import Number as mgp_Number
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from mgp import write_proc as mgp_write_proc

from mage.node2vec.graph import Graph, GraphHolder
from mage.node2vec.second_order_random_walk import SecondOrderRandomWalk


class Parameters:
    VECTOR_SIZE = "vector_size"
    WINDOW = "window"
    MIN_COUNT = "min_count"
    WORKERS = "workers"
    MIN_ALPHA = "min_alpha"
    SEED = "seed"
    ALPHA = "alpha"
    EPOCHS = "epochs"
    SG = "sg"
    HS = "hs"
    NEGATIVE = "negative"


NODE_EMBEDDING_PROPERTY = "embedding"


def learn_embeddings(
    walks: List[List[int]], **word2vec_params
) -> Dict[int, List[float]]:
    model = gensim_models.Word2Vec(sentences=walks, **word2vec_params)

    embeddings = {
        index: embedding
        for index, embedding in zip(
            model.wv.index_to_key, model.wv.vectors, strict=False
        )
    }

    return embeddings


def calculate_node_embeddings(
    graph: Graph,
    p: float,
    q: float,
    num_walks: int,
    walk_length: int,
    vector_size: int,
    alpha: float,
    window: int,
    min_count: int,
    seed: int,
    workers: int,
    min_alpha: float,
    sg: int,
    hs: int,
    negative: int,
    epochs: int,
) -> Dict[int, List[float]]:
    word2vec_params = {
        Parameters.VECTOR_SIZE: vector_size,
        Parameters.WINDOW: window,
        Parameters.MIN_COUNT: min_count,
        Parameters.WORKERS: workers,
        Parameters.MIN_ALPHA: min_alpha,
        Parameters.SEED: seed,
        Parameters.ALPHA: alpha,
        Parameters.EPOCHS: epochs,
        Parameters.SG: sg,
        Parameters.HS: hs,
        Parameters.NEGATIVE: negative,
    }

    second_order_random_walk = SecondOrderRandomWalk(
        p=p, q=q, num_walks=int(num_walks), walk_length=int(walk_length)
    )

    walks = second_order_random_walk.sample_node_walks(graph)
    embeddings = learn_embeddings(walks, **word2vec_params)
    return embeddings


def get_graph_memgraph_ctx(
    ctx: mgp_ProcCtx, edge_weight_property: str, is_directed: bool = False
) -> Graph:
    edges_weights = {}
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edge_weight = float(edge.properties.get(edge_weight_property, default=1))
            old_value = 0
            if (edge.from_vertex.id, edge.to_vertex.id) in edges_weights:
                old_value = edges_weights.get(
                    (edge.from_vertex.id, edge.to_vertex.id), ""
                )
            edges_weights[(edge.from_vertex.id, edge.to_vertex.id)] = (
                old_value + edge_weight
            )

    graph: Graph = GraphHolder(edges_weights, is_directed)
    return graph


@mgp_read_proc
def get_embeddings(
    ctx: mgp_ProcCtx,
    is_directed: bool = False,
    p=2.0,
    q=0.5,
    num_walks=4,
    walk_length=5,
    vector_size=100,
    alpha=0.025,
    window=5,
    min_count=1,
    seed=1,
    workers=1,
    min_alpha=0.0001,
    sg=1,
    hs=0,
    negative=5,
    epochs=5,
    edge_weight_property="weight",
) -> mgp_Record(nodes=mgp_List[mgp_Vertex], embeddings=mgp_List[mgp_List[mgp_Number]]):
    """
    Function to get node embeddings. Uses gensim.models.Word2Vec params.

    Parameters
    ----------
    is_directed : bool, optional
        If bool=True, graph is treated as directed, else not directed
    p : float, optional
        Return hyperparameter for calculating transition probabilities.
    q : float, optional
        Inout hyperparameter for calculating transition probabilities.
    num_walks : int, optional
        Number of walks per node in walk sampling.
    walk_length : int, optional
        Length of one walk in walk sampling.

    vector_size : int, optional
        Dimensionality of the word vectors.
    window : int, optional
        Maximum distance between the current and predicted word within a sentence.
    min_count : int, optional
        Ignores all words with total frequency lower than this.
    workers : int, optional
        Use these many worker threads to train the model (=faster training with multicore machines).
    sg : {0, 1}, optional
        Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs : {0, 1}, optional
        If 1, hierarchical softmax will be used for model training.
        If 0, and `negative` is non-zero, negative sampling will be used.
    negative : int, optional
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    cbow_mean : {0, 1}, optional
        If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    alpha : float, optional
        The initial learning rate.
    min_alpha : float, optional
        Learning rate will linearly drop to `min_alpha` as training progresses.
    seed : int, optional
        Seed for the random number generator. Initial vectors for each word are seeded with a hash of
        the concatenation of word + `str(seed)`.
    edge_weight_property: str,
        Property from graph in database from which you want to take edge weights.
    """
    graph: Graph = get_graph_memgraph_ctx(
        ctx=ctx, is_directed=is_directed, edge_weight_property=edge_weight_property
    )
    embeddings = calculate_node_embeddings(
        graph=graph,
        p=p,
        q=q,
        num_walks=num_walks,
        walk_length=walk_length,
        vector_size=vector_size,
        alpha=alpha,
        window=window,
        min_count=min_count,
        seed=seed,
        workers=workers,
        min_alpha=min_alpha,
        sg=sg,
        hs=hs,
        negative=negative,
        epochs=epochs,
    )

    embeddings_result = []
    nodes_result = []
    for node_id, embedding in embeddings.items():
        embeddings[node_id] = [float(e) for e in embedding]
        nodes_result.append(ctx.graph.get_vertex_by_id(node_id))
        embeddings_result.append(embeddings.get(node_id, False))
    # TODO (antoniofilipovic): when api becomes available, change to return list of records
    _return_value = mgp_Record(nodes=nodes_result, embeddings=embeddings_result)
    return _return_value


@mgp_write_proc
def set_embeddings(
    ctx: mgp_ProcCtx,
    is_directed: bool = False,
    p=2.0,
    q=0.5,
    num_walks=4,
    walk_length=5,
    vector_size=100,
    alpha=0.025,
    window=5,
    min_count=1,
    seed=1,
    workers=1,
    min_alpha=0.0001,
    sg=1,
    hs=0,
    negative=5,
    epochs=5,
    edge_weight_property="weight",
) -> mgp_Record(nodes=mgp_List[mgp_Vertex], embeddings=mgp_List[mgp_List[mgp_Number]]):
    """
    Function to get node embeddings. Uses gensim.models.Word2Vec params.

    Parameters
    ----------
    edges : List[mgp.Edge]
        All the edges in graph.
    is_directed : bool, optional
        If bool=True, graph is treated as directed, else not directed
    p : float, optional
        Return hyperparameter for calculating transition probabilities.
    q : float, optional
        Inout hyperparameter for calculating transition probabilities.
    num_walks : int, optional
        Number of walks per node in walk sampling.
    walk_length : int, optional
        Length of one walk in walk sampling.

    vector_size : int, optional
        Dimensionality of the word vectors.
    window : int, optional
        Maximum distance between the current and predicted word within a sentence.
    min_count : int, optional
        Ignores all words with total frequency lower than this.
    workers : int, optional
        Use these many worker threads to train the model (=faster training with multicore machines).
    sg : {0, 1}, optional
        Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs : {0, 1}, optional
        If 1, hierarchical softmax will be used for model training.
        If 0, and `negative` is non-zero, negative sampling will be used.
    negative : int, optional
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    cbow_mean : {0, 1}, optional
        If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    alpha : float, optional
        The initial learning rate.
    min_alpha : float, optional
        Learning rate will linearly drop to `min_alpha` as training progresses.
    seed : int, optional
        Seed for the random number generator. Initial vectors for each word are seeded with a hash of
        the concatenation of word + `str(seed)`.
    edge_weight_property: str,
        Property from graph in database from which you want to take edge weights.
    """
    graph: Graph = get_graph_memgraph_ctx(
        ctx=ctx, is_directed=is_directed, edge_weight_property=edge_weight_property
    )
    embeddings = calculate_node_embeddings(
        graph=graph,
        p=p,
        q=q,
        num_walks=num_walks,
        walk_length=walk_length,
        vector_size=vector_size,
        alpha=alpha,
        window=window,
        min_count=min_count,
        seed=seed,
        workers=workers,
        min_alpha=min_alpha,
        sg=sg,
        hs=hs,
        negative=negative,
        epochs=epochs,
    )

    embeddings_result = []
    nodes_result = []

    for node_id, embedding in embeddings.items():
        embeddings[node_id] = [float(e) for e in embedding]
        vertex = ctx.graph.get_vertex_by_id(node_id)
        vertex.properties.set(NODE_EMBEDDING_PROPERTY, embeddings.get(node_id, set()))

        nodes_result.append(ctx.graph.get_vertex_by_id(node_id))
        embeddings_result.append(embeddings.get(node_id, False))
    # TODO (antoniofilipovic): when api becomes available, change to return list of records
    _return_value = mgp_Record(nodes=nodes_result, embeddings=embeddings_result)
    return _return_value


@mgp_read_proc
def help() -> mgp_Record(name=str, value=str):
    """Shows manual page for node2vec"""
    records = []

    def make_records(name, doc):
        _return_value = (
            mgp_Record(name=n, value=v)
            for n, v in zip(
                chain([name], repeat("")), cleandoc(doc).splitlines(), strict=False
            )
        )
        return _return_value

    for func in (help, get_embeddings):
        records.extend(
            make_records("Procedure '{}'".format(func.__name__), func.__doc__)
        )

    return records
