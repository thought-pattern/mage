"""
Purpose of this query module is to offer easy kmeans clustering algorithm on top of the embeddings that you
might have stored in nodes. All you need to do is call kmeans.get_clusters(5, "embedding") where 5
represents number of clusters you want to get, and "embedding" represents node property name in which
embedding of node is stored
"""

from typing import List, Tuple

from mgp import Number as mgp_Number
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from mgp import write_proc as mgp_write_proc
from sklearn.cluster import KMeans


def get_created_clusters(
    number_of_clusters: int,
    embeddings: List[List[float]],
    nodes: List[mgp_Vertex],
    init: str,
    n_init: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    random_state: int,
) -> List[Tuple[mgp_Vertex, int]]:
    kmeans = KMeans(
        n_clusters=number_of_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        random_state=random_state,
    ).fit(embeddings)
    _return_value = [(nodes[i], label) for i, label in enumerate(kmeans.labels_)]
    return _return_value


def extract_nodes_embeddings(
    ctx: mgp_ProcCtx, embedding_property: str
) -> Tuple[List[mgp_Vertex], List[List[float]]]:
    nodes = []
    embeddings = []
    for node in ctx.graph.vertices:
        nodes.append(node)
        embeddings.append(node.properties.get(embedding_property, False))
    return nodes, embeddings


@mgp_read_proc
def get_clusters(
    ctx: mgp_ProcCtx,
    n_clusters: mgp_Number,
    embedding_property: str = "embedding",
    init: str = "k-means++",
    n_init: mgp_Number = 10,
    max_iter: mgp_Number = 10,
    tol: mgp_Number = 1e-4,
    algorithm: str = "lloyd",
    random_state: int = 1998,
) -> mgp_Record(node=mgp_Vertex, cluster_id=mgp_Number):
    nodes, embeddings = extract_nodes_embeddings(ctx, embedding_property)

    nodes_labels_list = get_created_clusters(
        number_of_clusters=n_clusters,
        embeddings=embeddings,
        nodes=nodes,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        random_state=random_state,
    )
    _return_value = [
        mgp_Record(node=node, cluster_id=int(label))
        for node, label in nodes_labels_list
    ]
    return _return_value


@mgp_write_proc
def set_clusters(
    ctx: mgp_ProcCtx,
    n_clusters: mgp_Number,
    embedding_property: str = "embedding",
    cluster_property="cluster_id",
    init: str = "k-means++",
    n_init: mgp_Number = 10,
    max_iter: mgp_Number = 10,
    tol: mgp_Number = 1e-4,
    algorithm: str = "lloyd",
    random_state=1998,
) -> mgp_Record(node=mgp_Vertex, cluster_id=mgp_Number):
    nodes, embeddings = extract_nodes_embeddings(ctx, embedding_property)

    nodes_labels_list = get_created_clusters(
        number_of_clusters=n_clusters,
        embeddings=embeddings,
        nodes=nodes,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        random_state=random_state,
    )

    for vertex, label in nodes_labels_list:
        vertex.properties.set(cluster_property, int(label))

    _return_value = [
        mgp_Record(node=node, cluster_id=int(label))
        for node, label in nodes_labels_list
    ]
    return _return_value
