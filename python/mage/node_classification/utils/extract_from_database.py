"""Utilities for extract from database."""

from collections import Counter

from numpy import add as np_add
from numpy import array as np_array
from numpy import shape as np_shape
from numpy import zeros as np_zeros
from torch import bool as torch_bool
from torch import float32 as torch_float32
from torch import long as torch_long
from torch import tensor as torch_tensor
from numpy.random import shuffle as np_shuffle
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData


def node_type_of(vertex) -> str:
    """Returns the node type key of a vertex: all of its labels joined by "_" in label order. Node stores, edge
    endpoint types and the reindexing maps all use this one encoding, so multi-label nodes and their edges agree."""
    if len(vertex.labels) == 0:
        raise ValueError(f"Node {vertex.id} has no labels.")
    node_type = "_".join(label.name for label in vertex.labels)
    return node_type


def nodes_fetching(
    nodes: list, features_name: str, class_name: str, data: HeteroData, observed_attribute: str = ""
) -> tuple[HeteroData, dict, dict, str, dict, dict]:
    """
    This procedure fetches the nodes from the database and returns
    them in HeteroData object.

    Args:
        nodes: The list of nodes from the database.
        features_name: The name of the features field.
        class_name: The name of the class field.
        data: HeteroData object where nodes will be stored.
        observed_attribute: The observed node type of an already trained task; empty to take the type of the first
            node with a class property.

    Returns:
        Tuple of (HeteroData, reindexing, inv_reindexing, observed_attribute).
    """

    # variable for storing node types
    node_types = []
    # variable for storing embedding lengths
    embedding_lengths = {}

    # class values of the observed type only; labels on other node types are not classes of this task
    observed_classes = set()

    for node in nodes:
        if features_name not in node.properties:
            continue  # if features are not available, skip the node

        # add node type to list of node types
        # (concatenate them if there are more than 1 label)
        node_type = node_type_of(node)
        node_types.append(node_type)

        # add embedding length to dictionary of embedding lengths
        if node_type not in embedding_lengths:
            embedding_lengths[node_type] = len(node.properties.get(features_name, []))

        # if observed attribute is not set, set it to node type
        if not observed_attribute and class_name in node.properties:
            observed_attribute = node_type

        if node_type == observed_attribute and class_name in node.properties:
            observed_classes.add(int(node.properties.get(class_name, 0)))

    # if node_types is empty, raise error
    if not node_types:
        raise Exception("There are no feature vectors found. Please check your database.")

    # Dense class vocabulary of the observed task, numbered in sorted class order so that it does not depend on graph
    # iteration order; the model's output width is the size of this vocabulary.
    label_reindexing = {label: index for index, label in enumerate(sorted(observed_classes))}
    inv_label_reindexing = {index: label for label, index in label_reindexing.items()}

    # apply Counter to obtain the number of each node type
    node_types = Counter(node_types)

    # auxiliary dictionaries for reindexing and inverse reindexing
    append_counter = {}
    reindexing = {}
    inv_reindexing = {}

    # since node_types is Counter, key is the node type and value is the number of nodes of that type
    for node_type, num_types_node in node_types.items():
        # for each node type, create a tensor of size num_types_node x embedding_lengths[node_type]
        data[node_type].x = torch_tensor(
            np_zeros((num_types_node, embedding_lengths.get(node_type, 0))),
            dtype=torch_float32,
        )

        # if node type is observed attribute, create other necessary tensors
        if node_type == observed_attribute:
            data[node_type].y = torch_tensor(np_zeros((num_types_node,), dtype=int), dtype=torch_long)

            data[node_type].train_mask = torch_tensor(np_zeros((num_types_node,), dtype=int), dtype=torch_bool)

            data[node_type].val_mask = torch_tensor(np_zeros((num_types_node,), dtype=int), dtype=torch_bool)

    # now fill the tensors with the nodes from the database
    for node in nodes:
        if features_name not in node.properties:
            continue  # if features are not available, skip the node

        node_type = node_type_of(node)

        node_type_counter = append_counter.get(node_type, 0)

        # add feature vector from database to tensor
        # it is checked at the start of the loop if features are available
        data[node_type].x[node_type_counter] = np_add(
            data[node_type].x[node_type_counter],
            np_array(node.properties.get(features_name, [])),
        )

        # store reindexing and inverse reindexing, installing each node type's maps on first use
        type_reindexing = reindexing.get(node_type, {})
        type_reindexing[node_type_counter] = node.id
        reindexing[node_type] = type_reindexing
        type_inv_reindexing = inv_reindexing.get(node_type, {})
        type_inv_reindexing[node.id] = node_type_counter
        inv_reindexing[node_type] = type_inv_reindexing

        # if node type is observed attribute, add classification label to tensor
        if node_type == observed_attribute:
            data[node_type].y[node_type_counter] = label_reindexing.get(int(node.properties.get(class_name, 0)), 0)

        # increase append_counter by 1
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
    """This procedure fetches the edges from the database and returns them in HeteroData object.

    Args:
        nodes: The list of nodes from the database.
        features_name: The name of the database features attribute.
        inv_reindexing: The inverse reindexing dictionary.
        data: HeteroData object where edges will be stored.

    Returns:
        HeteroData object with edges.
    """

    edges = []  # variable for storing edges
    edge_types = []  # variable for storing edge types
    append_counter = {}  # variable for storing append counter

    # obtain edges from context
    for vertex in nodes:
        for edge in vertex.out_edges:
            if features_name not in edge.from_vertex.properties or features_name not in edge.to_vertex.properties:
                continue  # if from_vertex or out_vertex do not have features, skip the edge

            # edge_type is (from_vertex type, edge name, to_vertex type), with endpoint types encoded as for nodes
            edge_type = (node_type_of(edge.from_vertex), edge.type.name, node_type_of(edge.to_vertex))

            edge_types.append(edge_type)  # append edge type to list of edge types
            edges.append(edge)  # append edge to list of edges

    edge_types = Counter(edge_types)  # apply Counter to obtain the number of each edge type

    # set edge_index variables to empty tensors of size 2 x no_edge_type_edges
    for edge_type, no_edge_type_edges in edge_types.items():
        data[edge_type].edge_index = torch_tensor(np_zeros((2, no_edge_type_edges)), dtype=torch_long)

    for edge in edges:
        from_vertex_type = node_type_of(edge.from_vertex)
        to_vertex_type = node_type_of(edge.to_vertex)
        edge_type = (from_vertex_type, edge.type.name, to_vertex_type)

        # every feature-bearing vertex was indexed by nodes_fetching under the same type key
        from_position = inv_reindexing.get(from_vertex_type, {}).get(edge.from_vertex.id, -1)
        to_position = inv_reindexing.get(to_vertex_type, {}).get(edge.to_vertex.id, -1)
        if from_position < 0 or to_position < 0:
            raise KeyError(f"Edge endpoint was not indexed as a {from_vertex_type}/{to_vertex_type} node")

        # add edge coordinates to edge_index tensors
        edge_position = append_counter.get(edge_type, 0)
        data[edge_type].edge_index[0][edge_position] = from_position
        data[edge_type].edge_index[1][edge_position] = to_position

        append_counter[edge_type] = edge_position + 1

    return data


def generating_masks_for_X(data: HeteroData, train_ratio: float, observed_attribute: str) -> HeteroData:
    """This procedure generates the masks for the nodes and edges.

    Args:
        data: HeteroData object with nodes and edges.

    Returns:
        HeteroData object with masks.
    """
    no_observed = np_shape(data[observed_attribute].x)[0]
    masks = np_zeros(no_observed)

    masks = np_add(
        masks,
        np_array(
            list(
                map(
                    lambda i: 1 if i < train_ratio * no_observed else 0,
                    range(no_observed),
                )
            )
        ),
    )

    np_shuffle(masks)

    data[observed_attribute].train_mask = torch_tensor(masks, dtype=torch_bool)

    data[observed_attribute].val_mask = torch_tensor(1 - masks, dtype=torch_bool)

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
    """This procedure extracts the data from the database and returns them in HeteroData object.

    Args:
        nodes: The list of nodes from the database.
        train_ratio: The ratio of training data.
        features_name: The name of the database features attribute.
        class_name: The name of the database class attribute.
        device: The device on which the data will be trained.
        observed_attribute: The observed node type of an already trained task (prediction on graphs that may carry
            no class labels); empty to derive it from the class property.
    """
    data = HeteroData()

    #################
    # NODES
    #################
    (
        data,
        reindexing,
        inv_reindexing,
        observed_attribute,
        label_reindexing,
        inv_label_reindexing,
    ) = nodes_fetching(nodes, features_name, class_name, data, observed_attribute)

    #################
    # EDGES
    #################
    data = edges_fetching(nodes, features_name, inv_reindexing, data)

    #################
    # MASKS
    #################
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
