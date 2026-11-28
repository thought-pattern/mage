"""Utilities for extract from database."""

from collections import Counter

from numpy import array as np_array, full as np_full, shape as np_shape, zeros as np_zeros
from numpy.random import shuffle as np_shuffle
from torch import (
    bool as torch_bool,
    float32 as torch_float32,
    long as torch_long,
    tensor as torch_tensor,
)
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData

UNLABELED_CLASS_INDEX = -1


def node_type_of(vertex) -> str:
    """Returns the node type key of a vertex: all of its labels joined by "_" in label order. Node stores, edge
    endpoint types and the reindexing maps all use this one encoding, so multi-label nodes and their edges agree.
    """
    if len(vertex.labels) == 0:
        raise ValueError(f"Node {vertex.id} has no labels.")
    node_type = "_".join(label.name for label in vertex.labels)
    return node_type


def nodes_fetching(
    nodes: list,
    features_name: str,
    class_name: str,
    data: HeteroData,
    observed_attribute: str = "",
) -> tuple[HeteroData, dict, dict, str, dict, dict]:
    """
    This procedure fetches the nodes from the database and returns
    them in HeteroData object.
    """
    node_types = []
    embedding_lengths = {}
    observed_classes = set()

    for node in nodes:
        if features_name not in node.properties:
            continue

        node_type = node_type_of(node)
        node_types.append(node_type)

        raw_features = node.properties.get(features_name, [])
        vec = np_array(raw_features, dtype=float)
        if vec.ndim != 1 or vec.shape[0] < 1:
            raise ValueError(f"Node {node.id} feature '{features_name}' must be a non-empty 1-D vector.")
        if node_type not in embedding_lengths:
            embedding_lengths[node_type] = int(vec.shape[0])
        elif int(vec.shape[0]) != embedding_lengths[node_type]:
            raise ValueError(
                f"Node {node.id} of type '{node_type}' has feature length {int(vec.shape[0])}, "
                f"expected {embedding_lengths[node_type]}."
            )

        if not observed_attribute and class_name in node.properties:
            observed_attribute = node_type

        if node_type == observed_attribute and class_name in node.properties:
            observed_classes.add(int(node.properties.get(class_name, 0)))

    if not node_types:
        raise ValueError("There are no feature vectors found. Please check your database.")

    label_reindexing = {label: index for index, label in enumerate(sorted(observed_classes))}
    inv_label_reindexing = {index: label for label, index in label_reindexing.items()}

    node_types_counts = Counter(node_types)

    append_counter = {}
    reindexing = {}
    inv_reindexing = {}

    for node_type, num_types_node in node_types_counts.items():
        data[node_type].x = torch_tensor(
            np_zeros((num_types_node, embedding_lengths.get(node_type, 0))),
            dtype=torch_float32,
        )

        if node_type == observed_attribute:
            data[node_type].y = torch_tensor(
                np_full((num_types_node,), UNLABELED_CLASS_INDEX, dtype=int),
                dtype=torch_long,
            )
            data[node_type].train_mask = torch_tensor(np_zeros((num_types_node,), dtype=int), dtype=torch_bool)
            data[node_type].val_mask = torch_tensor(np_zeros((num_types_node,), dtype=int), dtype=torch_bool)

    for node in nodes:
        if features_name not in node.properties:
            continue

        node_type = node_type_of(node)
        node_type_counter = append_counter.get(node_type, 0)

        vec = np_array(node.properties.get(features_name, []), dtype=float)
        data[node_type].x[node_type_counter] = torch_tensor(vec, dtype=torch_float32)

        type_reindexing = reindexing.get(node_type, {})
        type_reindexing[node_type_counter] = node.id
        reindexing[node_type] = type_reindexing
        type_inv_reindexing = inv_reindexing.get(node_type, {})
        type_inv_reindexing[node.id] = node_type_counter
        inv_reindexing[node_type] = type_inv_reindexing

        if node_type == observed_attribute and class_name in node.properties:
            raw_class = int(node.properties.get(class_name, 0))
            if raw_class in label_reindexing:
                data[node_type].y[node_type_counter] = label_reindexing.get(raw_class, UNLABELED_CLASS_INDEX)

        append_counter[node_type] = node_type_counter + 1

    return (
        data,
        reindexing,
        inv_reindexing,
        observed_attribute,
        label_reindexing,
        inv_label_reindexing,
    )


def edges_fetching(nodes: list, features_name: str, inv_reindexing: dict, data: HeteroData) -> HeteroData:
    """This procedure fetches the edges from the database and returns them in HeteroData object."""
    edges = []
    edge_types = []
    append_counter = {}

    for vertex in nodes:
        for edge in vertex.out_edges:
            if features_name not in edge.from_vertex.properties or features_name not in edge.to_vertex.properties:
                continue

            edge_type = (
                node_type_of(edge.from_vertex),
                edge.type.name,
                node_type_of(edge.to_vertex),
            )
            edge_types.append(edge_type)
            edges.append(edge)

    edge_types_counts = Counter(edge_types)

    for edge_type, no_edge_type_edges in edge_types_counts.items():
        data[edge_type].edge_index = torch_tensor(np_zeros((2, no_edge_type_edges)), dtype=torch_long)

    for edge in edges:
        from_vertex_type = node_type_of(edge.from_vertex)
        to_vertex_type = node_type_of(edge.to_vertex)
        edge_type = (from_vertex_type, edge.type.name, to_vertex_type)

        from_position = inv_reindexing.get(from_vertex_type, {}).get(edge.from_vertex.id, -1)
        to_position = inv_reindexing.get(to_vertex_type, {}).get(edge.to_vertex.id, -1)
        if from_position < 0 or to_position < 0:
            raise KeyError(f"Edge endpoint was not indexed as a {from_vertex_type}/{to_vertex_type} node")

        edge_position = append_counter.get(edge_type, 0)
        data[edge_type].edge_index[0][edge_position] = from_position
        data[edge_type].edge_index[1][edge_position] = to_position
        append_counter[edge_type] = edge_position + 1

    return data


def generating_masks_for_X(data: HeteroData, train_ratio: float, observed_attribute: str) -> HeteroData:
    """This procedure generates the masks for the nodes and edges."""
    if observed_attribute in data.node_types:
        no_observed = np_shape(data[observed_attribute].x)[0]
        labels = data[observed_attribute].y.detach().cpu().numpy().tolist()
        labeled_positions = [position for position, label in enumerate(labels) if label >= 0]
        labeled_count = len(labeled_positions)
        if labeled_count == 0:
            train_count = 0
        elif labeled_count == 1 or train_ratio >= 1.0:
            train_count = labeled_count
        elif train_ratio <= 0.0:
            train_count = 0
        else:
            train_count = min(labeled_count - 1, max(1, int(train_ratio * labeled_count)))

        split_flags = np_array(
            [1 if position < train_count else 0 for position in range(labeled_count)],
            dtype=int,
        )
        np_shuffle(split_flags)

        train_flags = np_zeros(no_observed, dtype=int)
        val_flags = np_zeros(no_observed, dtype=int)
        for node_position, is_train in zip(labeled_positions, split_flags.tolist(), strict=False):
            if is_train:
                train_flags[node_position] = 1
            else:
                val_flags[node_position] = 1

        data[observed_attribute].train_mask = torch_tensor(train_flags, dtype=torch_bool)
        data[observed_attribute].val_mask = torch_tensor(val_flags, dtype=torch_bool)

    data = T.AddSelfLoops()(data)
    data = T.ToUndirected()(data)

    return data


def extract_from_database(
    nodes: list,
    train_ratio: float,
    features_name: str,
    class_name: str,
    device: str,
    observed_attribute: str = "",
) -> tuple[HeteroData, str, dict, dict, dict, dict]:
    """This procedure extracts the data from the database and returns them in HeteroData object."""
    data = HeteroData()

    (
        data,
        reindexing,
        inv_reindexing,
        observed_attribute,
        label_reindexing,
        inv_label_reindexing,
    ) = nodes_fetching(nodes, features_name, class_name, data, observed_attribute)

    data = edges_fetching(nodes, features_name, inv_reindexing, data)
    data = generating_masks_for_X(data, train_ratio, observed_attribute)
    data = data.to(device, non_blocking=True)

    return (
        data,
        observed_attribute,
        reindexing,
        inv_reindexing,
        label_reindexing,
        inv_label_reindexing,
    )
