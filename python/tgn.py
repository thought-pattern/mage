(
    "\nThis module represents entry point to temporal graph networks Python implementation of Temp"  # Continue literal.
    "oral graph networks for\ndeep learning on dynamic graphs paper https://arxiv.org/pdf/2006.106"  # Continue literal.
    "37.pdf introduced by E.Rossi [erossi@twitter.com]\nduring his work in Twitter.\n\nTemporal grap"  # Continue literal.
    "h networks(TGNs) is a graph neural network method on dynamic graphs. In the recent years,\ngr"  # Continue literal.
    "aph neural networks have become very popular due to their ability to perform a wide variatio"  # Continue literal.
    "n of machine learning\ntasks on graphs, such as link prediction, node classification and so o"  # Continue literal.
    "n. This rise started with Graph convolutional\nnetworks introduced by Kipf et al, following b"  # Continue literal.
    "y GraphSAGE introduced by Hamilton et al, and in recent years new\nmethod was presented which"  # Continue literal.
    " introduces attention mechanism to graphs, known as Graph attention networks, by Veličković\n"  # Continue literal.
    "et al. Last two methods offer great possibility for inductive learning. But they haven't bee"  # Continue literal.
    "n specifically developed\nto handle different events occurring on graphs, such as node featur"  # Continue literal.
    "es update, node deletion, edge deletion and so on.\n\nIn his work, Rossi et al introduced to u"  # Continue literal.
    "s Temporal graph networks which present great possibility to do graph machine\nlearning on st"  # Continue literal.
    "ream of data, use-case occurring more often in recent years.\n\nWhat we have covered in this m"  # Continue literal.
    "odule\n  * **link prediction** - train your **TGN** to predict new **links/edges** and **node"  # Continue literal.
    " classification** - predict labels of nodes from graph structure and **node/edge** features\n"  # Continue literal.
    "  * **graph attention layer** embedding calculation and **graph sum layer** embedding layer "  # Continue literal.
    "calculation\n  * **mean** and **last** as message aggregators\n  * **mlp** and **identity(conc"  # Continue literal.
    "atenation)** as message functions\n  * **gru** and **rnn** as memory updater\n  * **uniform** "  # Continue literal.
    "temporal neighborhood sampler\n  * **memory** store and **raw message store**\n\nas introduced "  # Continue literal.
    "by **[Rossi et al](https://emanuelerossi.co.uk/)**.\n\nFollowing means **you** can use **TGN**"  # Continue literal.
    " to be able to **predict edges** or  to perform **node classification** tasks, with **graph "  # Continue literal.
    "attention layer** or **graph sum layer**, by using\neither **mean** or **last** as message ag"  # Continue literal.
    "gregator, **mlp** or **identity** as message function, and finally  **gru** or **rnn** as me"  # Continue literal.
    "mory updater.\n\n\n************************************************** IMPORTANT ***************"  # Continue literal.
    "***************************\n\nTo start exploring our module, jump to python/mage/tgn/definiti"  # Continue literal.
    "ons/instances.py and pick one of the implementations,\neither TGNEdgesSelfSupervised for self"  # Continue literal.
    " supervised learning on edges or TGNSupervised for supervised learning on nodes classificati"  # Continue literal.
    "on.\n\nEach of those methods consist of following steps you should look for in the module:\n * "  # Continue literal.
    "self._process_previous_batches - if you follow paper this will include calculating of messag"  # Continue literal.
    "es collected for each node\n    (python/mage/tgn/definitions/message_function.py), aggregatio"  # Continue literal.
    "n of messages (python/mage/tgn/definitions/message_aggregator.py)\n    and finally memory upd"  # Continue literal.
    "ate part in (python/mage/tgn/definitions/memory_updater.py)\n * self._get_graph_data -> after"  # Continue literal.
    "wards we create computation graph used by graph attention layer or graph sum layer\n *  self."  # Continue literal.
    "_process_current_batch -> final step includes processing of current batch, updating raw mess"  # Continue literal.
    "age store from\n interaction events, and preparing our **raw_message_store** for next batches"  # Continue literal.
    "\n\n\nWhat is **not** implemented in the module:\n  * **node update/deletion events** since they"  # Continue literal.
    " occur very rarely - although we have prepared a terrain\n  * **edge deletion** events\n  * **"  # Continue literal.
    "time projection** embedding calculation and **identity** embedding calculation since author "  # Continue literal.
    "mentions\n    they perform very poorly on all datasets - although it is trivial to add new la"  # Continue literal.
    "yer\n\n\nWhat we believe we did and author probably failed to do:\n * Embedding calculation seem"  # Continue literal.
    "s to be off in authors work on GitHub page: https://github.com/twitter-research/tgn.\n    One"  # Continue literal.
    " of our developers of module dropped an issue on GitHub page of twitter-research, but they s"  # Continue literal.
    "eem to be too busy,\n    can't blame them.\n    Problem seems that author doesn't use newly ca"  # Continue literal.
    "lculated embeddings in next layers, but instead uses raw features from 0th layer,\n    which "  # Continue literal.
    "according to paper is wrong.\n\n\n"
)

from dataclasses import astuple as dataclasses_astuple
from dataclasses import dataclass as dataclasses_dataclass
from enum import Enum as enum_Enum
from math import ceil
from time import time as time_time
from typing import Any, Dict, List, Set, Tuple

from mgp import Edge as mgp_Edge
from mgp import List as mgp_List
from mgp import Map as mgp_Map
from mgp import Number as mgp_Number
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from numpy import append as np_append
from numpy import array as np_array
from numpy import concatenate as np_concatenate
from numpy import empty as np_empty
from numpy import ones as np_ones
from numpy import random as np_random
from numpy import rint as np_rint
from numpy import sum as np_sum
from numpy import zeros as np_zeros
from torch import Tensor as torch_Tensor
from torch import cat as torch_cat
from torch import cuda as torch_cuda
from torch import device as torch_device
from torch import float as torch_float
from torch import nn
from torch import nn as torch_nn
from torch import no_grad as torch_no_grad
from torch import ones as torch_ones
from torch import optim as torch_optim
from torch import tensor as torch_tensor
from torch import zeros as torch_zeros

from mage.tgn.constants import (
    MemoryUpdaterType,
    MessageAggregatorType,
    MessageFunctionType,
    TGNLayerType,
)
from mage.tgn.definitions.instances import (
    TGNGraphAttentionEdgeSelfSupervised,
    TGNGraphAttentionSupervised,
    TGNGraphSumEdgeSelfSupervised,
    TGNGraphSumSupervised,
)
from mage.tgn.definitions.tgn import TGN
from mage.tgn.helper.simple_mlp import MLP

###################
# params and classes
##################


# params TGN must receive
class TGNParameters:
    NUM_OF_LAYERS = "num_of_layers"
    LAYER_TYPE = "layer_type"
    MEMORY_DIMENSION = "memory_dimension"
    TIME_DIMENSION = "time_dimension"
    NUM_EDGE_FEATURES = "num_edge_features"
    NUM_NODE_FEATURES = "num_node_features"
    MESSAGE_DIMENSION = "message_dimension"
    NUM_NEIGHBORS = "num_neighbors"
    EDGE_FUNCTION_TYPE = "edge_message_function_type"
    MESSAGE_AGGREGATOR_TYPE = "message_aggregator_type"
    MEMORY_UPDATER_TYPE = "memory_updater_type"
    NUM_ATTENTION_HEADS = "num_attention_heads"
    DEVICE = "device"


class OptimizerParameters:
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"


class MemgraphObjectsProperties:
    NODE_FEATURES_PROPERTY = "node_features_property"
    EDGE_FEATURES_PROPERTY = "edge_features_property"
    NODE_LABELS_PROPERTY = "node_label_property"


class OtherProperties:
    LEARNING_TYPE = "learning_type"
    DEVICE_TYPE = "device_type"
    BATCH_SIZE = "batch_size"


class LearningType(enum_Enum):
    Supervised = "supervised"
    SelfSupervised = "self_supervised"


class DeviceType(enum_Enum):
    CUDA = "cuda"
    CPU = "cpu"


class TGNMode(enum_Enum):
    Train = "train"
    Eval = "eval"


@dataclasses_dataclass
class QueryModuleTGN:
    config: Dict[str, Any]
    tgn: TGN
    criterion: nn.BCELoss
    optimizer: torch_optim.Adam
    device: torch_device
    m_loss: List[float]  # mean loss
    mlp: MLP
    tgn_mode: TGNMode
    learning_type: LearningType
    # used in negative sampling for self_supervised
    all_edges: Set[Tuple[int, int]]
    # to get all embeddings
    all_embeddings: Dict[int, np_array]
    results_per_epochs: Dict[int, List[mgp_Record]]
    current_epoch: int
    global_edge_count: int
    train_eval_index_split: int
    memgraph_objects_properties: Dict[str, Any]


@dataclasses_dataclass
class QueryModuleTGNBatch:
    current_batch_size: int
    sources: np_array
    destinations: np_array
    timestamps: np_array
    edge_idxs: np_array
    node_features: Dict[int, torch_Tensor]
    edge_features: Dict[int, torch_Tensor]
    batch_size: int
    labels: np_array


##############################
# global tgn training variables
##############################


query_module_tgn: QueryModuleTGN
query_module_tgn_batch: QueryModuleTGNBatch

##############################
# constants
##############################

EPOCH_START = 1

DEFINED_INPUT_TYPES = {
    "learning_type": str,
    "batch_size": int,
    TGNParameters.NUM_OF_LAYERS: int,
    TGNParameters.LAYER_TYPE: str,
    TGNParameters.MEMORY_DIMENSION: int,
    TGNParameters.TIME_DIMENSION: int,
    TGNParameters.NUM_EDGE_FEATURES: int,
    TGNParameters.NUM_NODE_FEATURES: int,
    TGNParameters.MESSAGE_DIMENSION: int,
    TGNParameters.NUM_NEIGHBORS: int,
    TGNParameters.EDGE_FUNCTION_TYPE: str,
    TGNParameters.MESSAGE_AGGREGATOR_TYPE: str,
    TGNParameters.MEMORY_UPDATER_TYPE: str,
}

DEFAULT_VALUES = {
    OtherProperties.LEARNING_TYPE: "self_supervised",
    OtherProperties.BATCH_SIZE: 200,
    TGNParameters.NUM_OF_LAYERS: 2,
    TGNParameters.LAYER_TYPE: "graph_attn",
    TGNParameters.MEMORY_DIMENSION: 100,
    TGNParameters.TIME_DIMENSION: 100,
    TGNParameters.NUM_EDGE_FEATURES: 50,
    TGNParameters.NUM_NODE_FEATURES: 50,
    TGNParameters.MESSAGE_DIMENSION: 100,
    TGNParameters.NUM_NEIGHBORS: 15,
    TGNParameters.EDGE_FUNCTION_TYPE: "identity",
    TGNParameters.MESSAGE_AGGREGATOR_TYPE: "last",
    TGNParameters.MEMORY_UPDATER_TYPE: "gru",
    TGNParameters.NUM_ATTENTION_HEADS: 1,
    MemgraphObjectsProperties.NODE_FEATURES_PROPERTY: "features",
    MemgraphObjectsProperties.EDGE_FEATURES_PROPERTY: "features",
    MemgraphObjectsProperties.NODE_LABELS_PROPERTY: "label",
    OptimizerParameters.LEARNING_RATE: 1e-4,
    OptimizerParameters.WEIGHT_DECAY: 5e-5,
    OtherProperties.DEVICE_TYPE: "cuda",
}


#############################
# global helpers
#############################
def update_epoch_counter() -> int:
    global query_module_tgn

    query_module_tgn.current_epoch += 1
    return query_module_tgn.current_epoch


def get_current_epoch() -> int:
    global query_module_tgn
    return query_module_tgn.current_epoch


def get_current_batch() -> int:
    global query_module_tgn
    _return_value = len(
        query_module_tgn.results_per_epochs.get(query_module_tgn.current_epoch, [])
    )
    return _return_value


def initialize_results_per_epoch(current_epoch: int) -> bool:
    global query_module_tgn
    query_module_tgn.results_per_epochs[current_epoch] = []
    return False


def set_global_edge_count(new_global_edge_count: int) -> int:
    global query_module_tgn
    query_module_tgn.global_edge_count = new_global_edge_count
    return query_module_tgn.global_edge_count


def set_current_batch_size(new_current_batch_size: int) -> int:
    global query_module_tgn_batch
    query_module_tgn_batch.current_batch_size = new_current_batch_size
    return query_module_tgn_batch.current_batch_size


def append_batch_record_curr_epoch(
    current_epoch: int, record: mgp_Record
) -> Dict[int, List[mgp_Record]]:
    global query_module_tgn
    assert current_epoch in query_module_tgn.results_per_epochs, (
        "Current epoch not defined"
    )
    query_module_tgn.results_per_epochs.get(current_epoch, []).append(record)
    return query_module_tgn.results_per_epochs


def get_output_records() -> List[mgp_Record]:
    global query_module_tgn, EPOCH_START
    output_records = []

    for i in range(EPOCH_START, len(query_module_tgn.results_per_epochs) + 1):
        output_records.extend(query_module_tgn.results_per_epochs.get(i, False))
    return output_records


def is_tgn_initialized() -> bool:
    global query_module_tgn
    if query_module_tgn.tgn is None:
        return False
    return True


def get_link_score(src_tensor: torch_Tensor, dest_tensor: torch_Tensor) -> torch_Tensor:
    global query_module_tgn
    # along columns
    x = torch_cat([src_tensor, dest_tensor], dim=1)
    _return_value = query_module_tgn.mlp(x).squeeze(dim=0)
    return _return_value


#####################################

# init function

#####################################


def set_tgn(
    learning_type: LearningType,
    device_type: DeviceType,
    tgn_config: Dict[str, Any],
    optimizer_config: Dict[str, Any],
    memgraph_objects_properties_config: Dict[str, Any],
) -> bool:
    global query_module_tgn, EPOCH_START

    if device_type == device_type.CUDA and torch_cuda.is_available():
        device = torch_device("cuda")
    else:
        device = torch_device("cpu")

    tgn_config[TGNParameters.DEVICE] = device

    if learning_type == LearningType.SelfSupervised:
        tgn, mlp = get_tgn_self_supervised(tgn_config, device)
    else:
        tgn, mlp = get_tgn_supervised(tgn_config, device)

    criterion = torch_nn.BCELoss()
    optimizer = torch_optim.Adam(
        tgn.parameters(),
        lr=optimizer_config.get(OptimizerParameters.LEARNING_RATE, False),
        weight_decay=optimizer_config.get(OptimizerParameters.WEIGHT_DECAY, False),
    )

    query_module_tgn = QueryModuleTGN(
        config=tgn_config,
        tgn=tgn,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        m_loss=[],
        mlp=mlp,
        tgn_mode=TGNMode.Train,  # we start in train mode
        learning_type=learning_type,
        all_edges=set(),
        all_embeddings={},
        results_per_epochs={},
        current_epoch=EPOCH_START,
        global_edge_count=0,
        train_eval_index_split=0,  # this number represent number of edges when set_eval function was called
        memgraph_objects_properties=memgraph_objects_properties_config,
    )
    return False


def get_tgn_self_supervised(
    config: Dict[str, Any], device: torch_device
) -> Tuple[TGN, MLP]:
    """
    Set parameters for self supervised learning. Here we try to predict edges.
    """

    if config.get(TGNParameters.LAYER_TYPE, False) == TGNLayerType.GraphSumEmbedding:
        tgn = TGNGraphSumEdgeSelfSupervised(**config).to(device)
    else:
        tgn = TGNGraphAttentionEdgeSelfSupervised(**config).to(device)

    # When TGN outputs embeddings for source nodes and destination nodes,
    # since we are working with edges and edge predictions we will concatenate their features together
    # and get prediction with MLP whether it is edge or it isn't
    # Other possibility would be Hadamard product
    # ( https://en.wikipedia.org/wiki/Hadamard_product_(matrices) it is a fancy name for dot product)
    # of source embeddings and destination embeddings, but we went with concat since author used concatenation
    # and not Hadamard product, in original implementation
    mlp_in_features_dim = (
        config.get(TGNParameters.MEMORY_DIMENSION, 0.0)
        + config.get(TGNParameters.NUM_NODE_FEATURES, 0.0)
    ) * 2

    mlp = MLP([mlp_in_features_dim, mlp_in_features_dim // 2, 1]).to(device=device)

    return tgn, mlp


def get_tgn_supervised(config: Dict[str, Any], device: torch_device) -> Tuple[TGN, MLP]:
    """ """

    if config.get(TGNParameters.LAYER_TYPE, False) == TGNLayerType.GraphSumEmbedding:
        tgn = TGNGraphSumSupervised(**config).to(device)
    else:
        tgn = TGNGraphAttentionSupervised(**config).to(device)

    mlp_in_features_dim = config.get(TGNParameters.MEMORY_DIMENSION, 0.0) + config.get(
        TGNParameters.NUM_NODE_FEATURES, 0.0
    )

    # used as probability calculator for label
    mlp = MLP([mlp_in_features_dim, 64, 1]).to(device=device)

    return tgn, mlp


#
# Helper functions
#


def sample_negative(negative_num: int) -> Tuple[np_array, np_array]:
    """
    Currently sampling of negative nodes is done in completely random fashion, and it is possible to sample
    source-dest pair that are real edges
    """
    global query_module_tgn
    all_edges = query_module_tgn.all_edges
    all_src = list(set([src for src, dest in all_edges]))
    all_dest = list(set([src for src, dest in all_edges]))

    _return_value = (
        np_random.choice(all_src, negative_num, replace=True),
        np_random.choice(all_dest, negative_num, replace=True),
    )
    return _return_value


def unpack_tgn_batch_data():
    global query_module_tgn_batch
    _return_value = dataclasses_astuple(query_module_tgn_batch)
    return _return_value


def update_mode_reset_grads_check_dims() -> bool:
    global query_module_tgn, query_module_tgn_batch

    if query_module_tgn.tgn_mode == TGNMode.Train:
        # set training mode
        query_module_tgn.tgn.train()

        query_module_tgn.optimizer.zero_grad()
        query_module_tgn.tgn.detach_tensor_grads()

        # todo add so that we only work with latest 128 neighbors
        # query_module_tgn.tgn.subsample_neighborhood()
    else:
        query_module_tgn.tgn.eval()

    (
        _,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        _,
        edge_features,
        _,
        labels,
    ) = unpack_tgn_batch_data()
    assert (
        len(sources)
        == len(destinations)
        == len(timestamps)
        == len(edge_features)
        == len(edge_idxs)
        == len(labels)
    ), "Batch size training error"
    return False


def update_embeddings(
    embeddings_source: np_array,
    embeddings_dest: np_array,
    sources: np_array,
    destinations: np_array,
) -> bool:
    global query_module_tgn
    for i, node in enumerate(sources):
        query_module_tgn.all_embeddings[node] = embeddings_source[i]

    for i, node in enumerate(destinations):
        query_module_tgn.all_embeddings[node] = embeddings_dest[i]
    return False


#
# Batch processing - self_supervised
#
def process_batch_self_supervised() -> float:
    """
    Uses sources, destinations, timestamps, edge_features and node_features from transactions.
    It is possible that current_batch_size is not always consistent, but it is always greater than minimum required.
    """
    global query_module_tgn, query_module_tgn_batch

    # do all necessary checks and updates of gradients
    update_mode_reset_grads_check_dims()

    (
        _,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        node_features,
        edge_features,
        _,
        _,
    ) = unpack_tgn_batch_data()

    current_batch_size = len(sources)
    negative_src, negative_dest = sample_negative(current_batch_size)

    graph_data = (
        sources,
        destinations,
        negative_src,
        negative_dest,
        timestamps,
        edge_idxs,
        edge_features,
        node_features,
    )

    embeddings, embeddings_negative = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    embeddings_source_neg = embeddings_negative[:current_batch_size]
    embeddings_dest_neg = embeddings_negative[current_batch_size:]

    # first row concatenation
    src_embeddings, dest_embeddings = (
        torch_cat([embeddings_source, embeddings_source_neg], dim=0),
        torch_cat([embeddings_dest, embeddings_dest_neg], dim=0),
    )
    # score shape = (num_positive_edges + num_negative_edges, 1) ->
    # num_positive_edges == num_negative_edges == current_batch_size
    score = get_link_score(src_embeddings, dest_embeddings)

    pos_score = score[:current_batch_size]
    neg_score = score[current_batch_size:]
    pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()

    if query_module_tgn.tgn_mode == TGNMode.Train:
        pos_label = torch_ones(
            current_batch_size, dtype=torch_float, device=query_module_tgn.device
        )
        neg_label = torch_zeros(
            current_batch_size, dtype=torch_float, device=query_module_tgn.device
        )
        # use reshape to get 1 dimension in every case
        loss = query_module_tgn.criterion(
            pos_prob.reshape((-1,)), pos_label
        ) + query_module_tgn.criterion(neg_prob.reshape((-1,)), neg_label)

        loss.backward()
        query_module_tgn.optimizer.step()
        query_module_tgn.m_loss.append(loss.item())
    pos_prob_cpu = pos_prob.reshape((-1,)).detach().cpu()
    neg_prob_cpu = neg_prob.reshape((-1,)).detach().cpu()
    # todo antoniofilipovic - update once we have logging API
    print(
        "POS PROB | NEG PROB",
        pos_prob_cpu,
        neg_prob_cpu,
    )
    pred_score = np_concatenate(
        [
            pos_prob_cpu.numpy(),
            neg_prob_cpu.numpy(),
        ]
    )
    true_label = np_concatenate(
        [np_ones(current_batch_size), np_zeros(current_batch_size)]
    )
    # rint - round int to nearest even number
    # x = 0, x <= 0.5
    # x = 1, x > 0.5
    # sum correct labels, divide by total number of labels
    precision = (
        np_sum(np_rint(true_label) == np_rint(pred_score)) * 1.0 / len(true_label)
    )

    # update embeddings to the newest ones that we can return on user request
    update_embeddings(
        embeddings_source.detach().cpu().numpy(),
        embeddings_dest.detach().cpu().numpy(),
        sources,
        destinations,
    )

    return precision


#
# Training - supervised
#


def process_batch_supervised() -> float:
    global query_module_tgn, query_module_tgn_batch

    # do all necessary checks and updates of gradients
    update_mode_reset_grads_check_dims()

    (
        _,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        node_features,
        edge_features,
        _,
        labels,
    ) = unpack_tgn_batch_data()

    current_batch_size = len(sources)

    graph_data = (
        sources,
        destinations,
        timestamps,
        edge_idxs,
        edge_features,
        node_features,
    )

    embeddings = query_module_tgn.tgn(graph_data)

    embeddings_source = embeddings[:current_batch_size]
    embeddings_dest = embeddings[current_batch_size:]

    x = torch_cat([embeddings_source, embeddings_dest], dim=0)  # along rows

    score = query_module_tgn.mlp(x).squeeze(dim=0)

    src_score = score[:current_batch_size]
    dest_score = score[current_batch_size:]

    src_prob, dest_prob = src_score.sigmoid(), dest_score.sigmoid()

    pred_score = np_concatenate(
        [
            (src_prob.squeeze()).detach().cpu().numpy(),
            (dest_prob.squeeze()).detach().cpu().numpy(),
        ]
    )
    true_label = np_concatenate([np_array(labels[:, 0]), np_array(labels[:, 1])])

    precision = (
        np_sum(np_rint(true_label) == np_rint(pred_score)) * 1.0 / len(true_label)
    )

    # update embeddings to newest ones that we can return on user request
    update_embeddings(
        embeddings_source.detach().cpu().numpy(),
        embeddings_dest.cpu().detach().cpu().numpy(),
        sources,
        destinations,
    )

    if query_module_tgn.tgn_mode == TGNMode.Eval:
        return precision

    # backprop only in case of training
    with torch_no_grad():
        src_label = torch_tensor(
            labels[:, 0], dtype=torch_float, device=query_module_tgn.device
        )
        dest_label = torch_tensor(
            labels[:, 1], dtype=torch_float, device=query_module_tgn.device
        )

    loss = query_module_tgn.criterion(
        src_prob.squeeze(), src_label.squeeze()
    ) + query_module_tgn.criterion(dest_prob.squeeze(), dest_label.squeeze())
    loss.backward()
    query_module_tgn.optimizer.step()
    query_module_tgn.m_loss.append(loss.item())

    return precision


def create_torch_tensor(feature: Tuple, num_features: int) -> torch_Tensor:
    if feature is None:
        np_feature = np_random.uniform(
            0, 1, num_features
        )  # uniformly sample features from 0 to 1
    else:
        np_feature = np_array(feature)
    _return_value = torch_tensor(
        np_feature,
        requires_grad=True,
        device=query_module_tgn.device,
        dtype=torch_float,
    )
    return _return_value


def parse_mgp_edges_into_tgn_batch(edges: mgp_List[mgp_Edge]) -> QueryModuleTGNBatch:
    global query_module_tgn_batch, query_module_tgn

    for edge in edges:
        source = edge.from_vertex
        src_id = edge.from_vertex.id

        dest = edge.to_vertex
        dest_id = int(edge.to_vertex.id)

        # maybe this is not best practice, but since we are calling this
        # function only for processing of edges, we can also
        # update all edges to for negative sampling later used later on
        query_module_tgn.all_edges.add((src_id, dest_id))

        node_features_property = query_module_tgn.memgraph_objects_properties.get(
            MemgraphObjectsProperties.NODE_FEATURES_PROPERTY, False
        )
        edge_features_property = query_module_tgn.memgraph_objects_properties.get(
            MemgraphObjectsProperties.EDGE_FEATURES_PROPERTY, False
        )
        node_label_property = query_module_tgn.memgraph_objects_properties.get(
            MemgraphObjectsProperties.NODE_LABELS_PROPERTY, False
        )
        src_features = source.properties.get(node_features_property, False)
        dest_features = dest.properties.get(node_features_property, False)

        src_label = source.properties.get(node_label_property, 0)
        dest_label = dest.properties.get(node_label_property, 0)

        timestamp = edge.id
        edge_idx = int(edge.id)

        edge_feature = edge.properties.get(edge_features_property, False)

        query_module_tgn_batch.node_features[src_id] = create_torch_tensor(
            src_features,
            query_module_tgn.config.get(TGNParameters.NUM_NODE_FEATURES, False),
        )
        query_module_tgn_batch.node_features[dest_id] = create_torch_tensor(
            dest_features,
            query_module_tgn.config.get(TGNParameters.NUM_NODE_FEATURES, False),
        )
        query_module_tgn_batch.edge_features[edge_idx] = create_torch_tensor(
            edge_feature,
            query_module_tgn.config.get(TGNParameters.NUM_EDGE_FEATURES, False),
        )

        query_module_tgn_batch.sources = np_append(
            query_module_tgn_batch.sources, src_id
        )
        query_module_tgn_batch.destinations = np_append(
            query_module_tgn_batch.destinations, dest_id
        )
        query_module_tgn_batch.timestamps = np_append(
            query_module_tgn_batch.timestamps, timestamp
        )
        query_module_tgn_batch.edge_idxs = np_append(
            query_module_tgn_batch.edge_idxs, edge_idx
        )
        query_module_tgn_batch.labels = np_append(
            query_module_tgn_batch.labels, np_array([[src_label, dest_label]]), axis=0
        )
    return query_module_tgn_batch


def reset_tgn_batch(batch_size: int) -> bool:
    global query_module_tgn_batch
    query_module_tgn_batch = QueryModuleTGNBatch(
        0,
        np_empty((0, 1), dtype=int),
        np_empty((0, 1), dtype=int),
        np_empty((0, 1), dtype=int),
        np_empty((0, 1), dtype=int),
        {},
        {},
        batch_size,
        np_empty((0, 2), dtype=int),
    )
    return False


def reset_tgn() -> bool:
    global query_module_tgn

    # reset whole tgn
    query_module_tgn.all_embeddings = {}
    reset_tgn_batch(0)
    query_module_tgn.all_edges = set()
    return False


def process_epoch_batch() -> mgp_Record:
    batch_start_time = time_time()
    precision = (
        process_batch_self_supervised()
        if query_module_tgn.learning_type == LearningType.SelfSupervised
        else process_batch_supervised()
    )
    batch_process_time = time_time() - batch_start_time

    record = mgp_Record(
        epoch_num=get_current_epoch(),
        batch_num=get_current_batch() + 1,  # this is a new record batch
        batch_process_time=round(batch_process_time, 2),
        precision=round(precision, 2),
        batch_type=query_module_tgn.tgn_mode.name,
    )

    # add same logging as in core
    print(
        f"EPOCH {get_current_epoch()} || BATCH {get_current_batch()}, | batch_process_time="
        f"{batch_process_time}  | precision={precision}"
    )

    return record


def train_eval_epochs(
    num_epochs: int, train_edges: List[mgp_Edge], eval_edges: List[mgp_Edge]
) -> bool:
    global query_module_tgn, query_module_tgn_batch

    batch_size = query_module_tgn_batch.batch_size
    num_train_edges = len(train_edges)
    num_train_batches = ceil(num_train_edges / batch_size)

    num_eval_edges = len(eval_edges)
    num_eval_batches = ceil(num_eval_edges / batch_size)
    assert batch_size > 0

    for _epoch in range(num_epochs):
        # update global epoch counter
        update_epoch_counter()

        # initialize container for current epochs where we save records for
        # precision results and batch processing time
        initialize_results_per_epoch(get_current_epoch())

        # on every epoch training tgn should have clean state,
        # although we update parameters, memory should be empty,
        # temporal neighborhood should be empty and message store should be empty
        # because if it isn't we have problem with information leakage from future
        # to current training samples
        query_module_tgn.tgn.init_memory()
        query_module_tgn.tgn.init_temporal_neighborhood()
        query_module_tgn.tgn.init_message_store()

        query_module_tgn.all_edges = set()
        query_module_tgn.m_loss = []

        reset_tgn_batch(batch_size=batch_size)

        # when we update here tgn_mode, later when we call process_batch_self_supervised
        # it will change tgn mode to .train() so no worries
        query_module_tgn.tgn_mode = TGNMode.Train

        for i in range(num_train_batches):
            # sample edges we need
            start_index_train_batch = i * batch_size
            end_index_train_batch = min((i + 1) * batch_size, num_train_edges)
            current_edges_batch = train_edges[
                start_index_train_batch:end_index_train_batch
            ]

            # prepare batch
            parse_mgp_edges_into_tgn_batch(current_edges_batch)
            batch_result_record = process_epoch_batch()
            append_batch_record_curr_epoch(get_current_epoch(), batch_result_record)

            # reset for next batch
            reset_tgn_batch(batch_size=batch_size)

        # here we need to change mode to eval
        # also later when we call process_batch_self_supervised
        # it will change tgn mode to .eval()
        query_module_tgn.tgn_mode = TGNMode.Eval
        for i in range(num_eval_batches):
            # sample edges we need
            start_index_eval_batch = i * batch_size
            end_index_eval_batch = min((i + 1) * batch_size, num_eval_edges)
            current_edges_batch = eval_edges[
                start_index_eval_batch:end_index_eval_batch
            ]
            # prepare batch
            parse_mgp_edges_into_tgn_batch(current_edges_batch)
            batch_result_record = process_epoch_batch()
            append_batch_record_curr_epoch(get_current_epoch(), batch_result_record)

            # reset for next batch
            reset_tgn_batch(batch_size=batch_size)
    return False


#####################################################

# all available read_procs


#####################################################
@mgp_read_proc
def predict_link_score(
    ctx: mgp_ProcCtx, src: mgp_Vertex, dest: mgp_Vertex
) -> mgp_Record(prediction=mgp_Number):
    """
    If you were doing link prediction, with this function you can input some of your vertices, and get the predictions
    Be careful to input vertices in correct order (src->dest) otherwise you might get wrong prediction

    :params src: src vertex in prediction
    :params dest: dest vertex in prediction

    :return prediction: score between 0 and 1
    """
    global query_module_tgn

    embedding_source: List[float] = query_module_tgn.all_embeddings.get(int(src.id), [])
    embedding_dest = query_module_tgn.all_embeddings.get(int(dest.id), False)

    embedding_src_torch = torch_tensor(
        embedding_source, device=query_module_tgn.device, dtype=torch_float
    ).reshape(1, -1)
    embedding_dest_torch = torch_tensor(
        embedding_dest, device=query_module_tgn.device, dtype=torch_float
    ).reshape(1, -1)

    # column concatenation
    score = get_link_score(embedding_src_torch, embedding_dest_torch)
    _return_value = mgp_Record(prediction=float(score))
    return _return_value


@mgp_read_proc
def train_and_eval(ctx: mgp_ProcCtx, num_epochs: int) -> mgp_Record(
    epoch_num=mgp_Number,
    batch_num=mgp_Number,
    batch_process_time=mgp_Number,
    precision=mgp_Number,
    batch_type=str,
):
    """
    After calling this function from ctx we will get all edges currently in database, split them in ratio of
    train and eval edges if function set_mode("eval") was called at some point and use training edges to train
    further TGN and eval edges to evaluate our model.

    If you didn't call function set_mode("eval"), you won't be able to do train and eval.

    :param num_epochs: number of epochs used for training and evaluation
    :train_eval_percent_split: dataset split ratio on train and eval


    :return: mgp.Record(): empty record if everything was fine
    """

    global query_module_tgn

    if not is_tgn_initialized():
        raise Exception(
            "TGN is not initialized still. Call `set_params` function in order to initialize it."
        )

    vertices = ctx.graph.vertices
    curr_all_edges = []
    for vertex in vertices:
        curr_all_edges.extend(list(vertex.out_edges))

    curr_all_edges = sorted(curr_all_edges, key=lambda x: x.id)

    # note: if you didn't call mode switch to eval, you can't
    # still do epoch training
    if query_module_tgn.train_eval_index_split == 0:
        raise Exception(
            "Can't call train and eval if you didn't change TGN mode to 'eval'"
        )

    train_eval_epochs(
        num_epochs=num_epochs,
        train_edges=curr_all_edges[: query_module_tgn.train_eval_index_split],
        eval_edges=curr_all_edges[query_module_tgn.train_eval_index_split :],
    )
    # get all records for every epoch and every batch inside it as results
    _return_value = get_output_records()
    return _return_value


@mgp_read_proc
def get_results(
    ctx: mgp_ProcCtx,
) -> mgp_Record(
    epoch_num=mgp_Number,
    batch_num=mgp_Number,
    batch_process_time=mgp_Number,
    precision=mgp_Number,
    batch_type=str,
):
    """
    This method returns all results from training and evaluation on all epochs

    :return: mgp.Record(): List of records of training and evaluation stats

    """
    # get all records for every epoch and every batch inside it as results
    if not is_tgn_initialized():
        raise Exception(
            "TGN is not initialized still. Call `set_params` function in order to initialize it."
        )

    _return_value = get_output_records()
    return _return_value


@mgp_read_proc
def set_eval(ctx: mgp_ProcCtx) -> mgp_Record(message=str):
    """
    Purpose of this function is to switch mode from "train" to "eval" at some point during your stream.
    At that point, we will save current edge count, and this information will later be used in function
    `train_and_eval` to split edges from Memgraph in train and eval set

    :return: mgp.Record(): empty record if everything was fine
    """
    global query_module_tgn

    if not is_tgn_initialized():
        raise Exception(
            "TGN is not initialized still. Call `set_params` function in order to initialize it."
        )

    query_module_tgn.train_eval_index_split = query_module_tgn.global_edge_count
    query_module_tgn.tgn_mode = TGNMode.Eval

    _return_value = mgp_Record(message="TGN mode changed to 'eval'.")
    return _return_value


@mgp_read_proc
def revert_from_database(ctx: mgp_ProcCtx) -> mgp_Record():
    """
    todo implement
    Revert from database and potential file in var/log/ to which we can save params
    """
    raise NotImplementedError(
        "You can check what is implemented at our docs page: https://memgraph.com/docs/mage"
    )


@mgp_read_proc
def save_tgn_params(ctx: mgp_ProcCtx) -> mgp_Record():
    """
    todo implement
    After every batch we could add saving params as checkpoints to var/log/memgraph
    This is how it is done usually in ML
    """
    raise NotImplementedError(
        "You can check what is implemented at our docs page: https://memgraph.com/docs/mage"
    )


@mgp_read_proc
def reset(ctx: mgp_ProcCtx) -> mgp_Record(message=str):
    reset_tgn()
    _return_value = mgp_Record(message="Reset was successful.")
    return _return_value


@mgp_read_proc
def get(ctx: mgp_ProcCtx) -> mgp_Record(node=mgp_Vertex, embedding=mgp_List[float]):
    """
    Get all embeddings of nodes created by TGN. These are final embeddings, after num_layers of processing

    One note here: embeddings are not for same timestamp, since one node could have last interaction at time
    t1 and current timestamp can be tn, where t1<tn

    We can't update embedding of node if it doesn't have any interactions, but we can update node's memory, so next
    time it appears in some interaction event, it won't suffer from memory staleness problem as mentioned in original
    paper

    :param edges: list of edges to preprocess, and if current batch size is big enough use for training or evaluation

    :return: mgp.Record(): empty record if everything was fine
    """
    global query_module_tgn

    if not is_tgn_initialized():
        raise Exception(
            "TGN is not initialized still. Call `set_params` function in order to initialize it."
        )

    embeddings_dict = {}

    for node_id, embedding in query_module_tgn.all_embeddings.items():
        embeddings_dict[node_id] = [float(e) for e in embedding]

    _return_value = [
        mgp_Record(node=ctx.graph.get_vertex_by_id(node_id), embedding=embedding)
        for node_id, embedding in embeddings_dict.items()
    ]
    return _return_value


@mgp_read_proc
def update(ctx: mgp_ProcCtx, edges: mgp_List[mgp_Edge]) -> mgp_Record():
    (
        "\n    Purpose of following function is to process edges which are created in Memgraph, and ge"  # Continue literal.
        "t features from\n    nodes or edges if they are present and save all that data so when batch "  # Continue literal.
        'is greater than\n    predefined size `batch_size` saved in object `query_mode_tgn` "set_param'  # Continue literal.
        's",\n    this function will call training on those edges of TGN module. This represents one b'  # Continue literal.
        'atch\n    and until you call tgn.set_mode("eval") all those edges will be used as training ed'  # Continue literal.
        'ges.\n\n    If you have a stream of data and you expect let\'s say 20000 edges, you can "split"'  # Continue literal.
        ' your stream of data\n    in "train" set and "eval" set by calling method tgn.set_mode("eval"'  # Continue literal.
        ").\n\n\n    Once again you should not call this function yourself but set trigger\n    on --> CR"  # Continue literal.
        "EATE event (edge create event), and Memgraph will give edges as input\n\n    Your toy example "  # Continue literal.
        "could look like following\n\n    CREATE INDEX ON :User(id);\n    CREATE INDEX ON :Item(id);\n   "  # Continue literal.
        ' CALL tgn.set_params("self_supervised", 200, 2, "graph_attn", 100, 100, 20, 20, 100, 10, "id'  # Continue literal.
        'entity", "last", "gru", 1) YIELD *;\n    MERGE (a:User {id: \'A1BHUGKLYW6H7V\', profile_name:\'P'  # Continue literal.
        ". Lecuyer'}) MERGE (b:Item {id: 'B0007MCVQ2'}) MERGE (a)-[:REVIEWED { features: [161.0,..., "  # Continue literal.
        '0.9238], ...}]->(b);\n    Here create more edges\n    CALL tgn.set_mode("eval") YIELD *;\n    H'  # Continue literal.
        "ere create some edges used for evaluation\n    CALL tgn.train_and_eval(5) YIELD * RETURN *;\n\n"  # Continue literal.
        '    This way all edges **until** you call `CALL tgn.set_mode("eval") YIELD *;` will be used '  # Continue literal.
        "for **training**,\n    and all edges after such call will be used for **evaluation**.\n\n    Af"  # Continue literal.
        "ter you make a query `CALL tgn.train_and_eval(5) YIELD * RETURN *;`, we will get all edges f"  # Continue literal.
        'rom our database,\n    and because you called tgn.set_mode("eval") at some point, we save at '  # Continue literal.
        "which point it happened at we will split\n    train and eval edges in same manner.\n\n    :para"  # Continue literal.
        "m edges: list of edges to preprocess, and if current batch size is big enough use for traini"  # Continue literal.
        "ng or evaluation\n\n    :return: mgp.Record(): empty record if everything was fine\n"
    )
    global query_module_tgn_batch, query_module_tgn

    if not is_tgn_initialized():
        raise Exception(
            "TGN is not initialized still. Call `set_params` function in order to initialize it."
        )

    num_edges = len(edges)

    # we track number of edges so
    set_global_edge_count(query_module_tgn.global_edge_count + num_edges)

    # update current batch size with new edges
    set_current_batch_size(query_module_tgn_batch.current_batch_size + num_edges)

    # we update our batch with current edges
    parse_mgp_edges_into_tgn_batch(edges)
    # if batch is still not full, we don't go to "train" or "eval" of TGN
    if query_module_tgn_batch.current_batch_size < query_module_tgn_batch.batch_size:
        _return_value = mgp_Record()
        return _return_value
    # this is just check if we have initialized list to save records of batches for training or evaluation
    if get_current_epoch() not in query_module_tgn.results_per_epochs:
        initialize_results_per_epoch(get_current_epoch())

    # process epoch in self_supervised or supervised mode in a given mode which
    # can be "train" or "eval"
    batch_result_record = process_epoch_batch()

    append_batch_record_curr_epoch(get_current_epoch(), batch_result_record)

    # reset for next batch
    reset_tgn_batch(batch_size=query_module_tgn_batch.batch_size)

    _return_value = mgp_Record()
    return _return_value


@mgp_read_proc
def set_params(
    ctx: mgp_ProcCtx,
    params: mgp_Map,
) -> mgp_Record():
    (
        "\n    With following function you can define parameters used in TGN, as well as what kind of "  # Continue literal.
        'learning you want\n    to do with TGN module.\n\n    If you set TGN to "self_supervised" mode, '  # Continue literal.
        'it will try to predict new edges, and with "supervised" mode it will try\n    to predict labe'  # Continue literal.
        "ls\n    How to call this method:\n        CALL tgn.set_params({learning_type:'self_supervised'"  # Continue literal.
        ", batch_size:200, num_of_layers:2, layer_type:'graph_attn', memory_dimension:20, time_dimens"  # Continue literal.
        "ion:50, num_edge_features:20,\n        num_node_features:20, message_dimension:100, num_neigh"  # Continue literal.
        "bors:15, edge_message_function_type:'identity',message_aggregator_type:'last', memory_update"  # Continue literal.
        "r_type:'gru', num_attention_heads:1});\n    Warning: Every time you call this function, old T"  # Continue literal.
        "GN object is cleared and process of learning params is\n    restarted\n\n    :param params: Dic"  # Continue literal.
        't containing following keys:\n        learning_type: "self_supervised" or "supervised" depend'  # Continue literal.
        "ing on if you want to predict edges or node labels\n        batch_size: size of batch to proc"  # Continue literal.
        "ess by TGN, recommended size 200\n        num_of_layers: number of layers of graph neural net"  # Continue literal.
        'work, 2 is optimal size, GNNs perform worse with bigger size\n        layer_type: "graph_attn'  # Continue literal.
        '" or "graph_sum" layer type as defined in original paper\n        memory_dimension: dimension'  # Continue literal.
        ' of memory tensor of each node\n        time_dimension: dimension of time vector from "time2v'  # Continue literal.
        'ec" paper\n        num_edge_features: number of edge features we will use from each edge\n    '  # Continue literal.
        "    num_node_features: number of expected node features\n        message_dimension: dimension"  # Continue literal.
        " of message, only used if you use MLP as message function type, otherwise ignored\n        nu"  # Continue literal.
        "m_neighbors: number of sampled neighbors\n        edge_message_function_type: message functio"  # Continue literal.
        'n type, "identity" for concatenation or "mlp" for projection\n        message_aggregator_type'  # Continue literal.
        ': message aggregator type, "mean" or "last"\n        memory_updater_type: memory updater type'  # Continue literal.
        ', "gru" or "rnn"\n        [optional] num_attention_heads: number of attention heads used if *'  # Continue literal.
        '*only** if you define "graph_attn" as layer type\n        [optional] learning_rate: learning '  # Continue literal.
        "rate for optimizer\n        [optional] weight_decay: weight decay used in optimizer\n        ["  # Continue literal.
        "optional] device_type: type of device you want to use for training - cuda or cpu\n        [op"  # Continue literal.
        "tional] node_features_property: name of features property on nodes from which we read featur"  # Continue literal.
        "es\n        [optional] edge_features_property: name of features property on edges from which "  # Continue literal.
        "we read features\n        [optional] node_label_property: name of label property on nodes fro"  # Continue literal.
        "m which we read features\n\n    :return: mgp.Record(): empty record if everything was fine\n"
    )
    global query_module_tgn_batch, DEFINED_INPUT_TYPES, DEFAULT_VALUES

    # function checks if input values in dictionary are correctly typed
    def is_correctly_typed(defined_types, input_values):
        if isinstance(defined_types, dict) and isinstance(input_values, dict):
            # defined_types is a dict of types
            _return_value = all(
                k in input_values  # check if exists
                and is_correctly_typed(
                    defined_types[k], input_values[k]
                )  # check for correct type
                for k in defined_types
            )
            return _return_value
        elif isinstance(defined_types, type):
            _return_value = isinstance(input_values, defined_types)
            return _return_value
        else:
            return False

    params = {**DEFAULT_VALUES, **params}  # override any default parameters
    print(params)
    if not is_correctly_typed(DEFINED_INPUT_TYPES, params):
        raise Exception(
            f"Input dictionary is not correctly typed. Expected following types {DEFINED_INPUT_TYPES}."
        )

    learning_type: str = params.get(OtherProperties.LEARNING_TYPE, "")
    batch_size: int = params.get(OtherProperties.BATCH_SIZE, 0)

    reset_tgn_batch(batch_size)

    tgn_config = {
        TGNParameters.NUM_OF_LAYERS: params.get(TGNParameters.NUM_OF_LAYERS, False),
        TGNParameters.MEMORY_DIMENSION: params.get(
            TGNParameters.MEMORY_DIMENSION, False
        ),
        TGNParameters.TIME_DIMENSION: params.get(TGNParameters.TIME_DIMENSION, False),
        TGNParameters.NUM_EDGE_FEATURES: params.get(
            TGNParameters.NUM_EDGE_FEATURES, False
        ),
        TGNParameters.NUM_NODE_FEATURES: params.get(
            TGNParameters.NUM_NODE_FEATURES, False
        ),
        TGNParameters.MESSAGE_DIMENSION: params.get(
            TGNParameters.MESSAGE_DIMENSION, False
        ),
        TGNParameters.NUM_NEIGHBORS: params.get(TGNParameters.NUM_NEIGHBORS, False),
        TGNParameters.LAYER_TYPE: get_tgn_layer_enum(
            params.get(TGNParameters.LAYER_TYPE, False)
        ),
        TGNParameters.EDGE_FUNCTION_TYPE: get_edge_message_function_type(
            params.get(TGNParameters.EDGE_FUNCTION_TYPE, False)
        ),
        TGNParameters.MESSAGE_AGGREGATOR_TYPE: get_message_aggregator_type(
            params.get(TGNParameters.MESSAGE_AGGREGATOR_TYPE, False)
        ),
        TGNParameters.MEMORY_UPDATER_TYPE: get_memory_updater_type(
            params.get(TGNParameters.MEMORY_UPDATER_TYPE, False)
        ),
    }
    memgraph_objects_property_config = {
        MemgraphObjectsProperties.NODE_FEATURES_PROPERTY: params.get(
            MemgraphObjectsProperties.NODE_FEATURES_PROPERTY, False
        ),
        MemgraphObjectsProperties.EDGE_FEATURES_PROPERTY: params.get(
            MemgraphObjectsProperties.EDGE_FEATURES_PROPERTY, False
        ),
        MemgraphObjectsProperties.NODE_LABELS_PROPERTY: params.get(
            MemgraphObjectsProperties.NODE_LABELS_PROPERTY, False
        ),
    }

    optimizer_config = {
        OptimizerParameters.LEARNING_RATE: params.get(
            OptimizerParameters.LEARNING_RATE, False
        ),
        OptimizerParameters.WEIGHT_DECAY: params.get(
            OptimizerParameters.WEIGHT_DECAY, False
        ),
    }
    # tgn params

    if (
        tgn_config.get(TGNParameters.LAYER_TYPE, False)
        == TGNLayerType.GraphAttentionEmbedding
    ):
        tgn_config[TGNParameters.NUM_ATTENTION_HEADS] = params.get(
            TGNParameters.NUM_ATTENTION_HEADS, 1
        )

    # set learning type
    tgn_learning_type = get_learning_type(learning_type)
    tgn_device_type = get_device_type(params.get(OtherProperties.DEVICE_TYPE, False))

    set_tgn(
        tgn_learning_type,
        tgn_device_type,
        tgn_config,
        optimizer_config,
        memgraph_objects_property_config,
    )

    _return_value = mgp_Record()
    return _return_value


#####################################

# helper functions


#####################################
def get_tgn_layer_enum(layer_type: str) -> TGNLayerType:
    if TGNLayerType(layer_type) is TGNLayerType.GraphAttentionEmbedding:
        return TGNLayerType.GraphAttentionEmbedding
    elif TGNLayerType(layer_type) is TGNLayerType.GraphSumEmbedding:
        return TGNLayerType.GraphSumEmbedding
    else:
        raise Exception(
            f"Wrong layer type, expected {TGNLayerType.GraphAttentionEmbedding} or {TGNLayerType.GraphSumEmbedding} "
        )


def get_edge_message_function_type(message_function_type: str) -> MessageFunctionType:
    if MessageFunctionType(message_function_type) is MessageFunctionType.Identity:
        return MessageFunctionType.Identity
    elif MessageFunctionType(message_function_type) is MessageFunctionType.MLP:
        return MessageFunctionType.MLP
    else:
        raise Exception(
            f"Wrong message function type, expected {MessageFunctionType.Identity} or {MessageFunctionType.MLP} "
        )


def get_message_aggregator_type(message_aggregator_type: str) -> MessageAggregatorType:
    if MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Mean:
        return MessageAggregatorType.Mean
    elif MessageAggregatorType(message_aggregator_type) is MessageAggregatorType.Last:
        return MessageAggregatorType.Last
    else:
        raise Exception(
            f"Wrong message aggregator type, expected {MessageAggregatorType.Last} or {MessageAggregatorType.Mean} "
        )


def get_memory_updater_type(memory_updater_type: str) -> MemoryUpdaterType:
    if MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.GRU:
        return MemoryUpdaterType.GRU

    elif MemoryUpdaterType(memory_updater_type) is MemoryUpdaterType.RNN:
        return MemoryUpdaterType.RNN

    else:
        raise Exception(
            f"Wrong memory updater type, expected {MemoryUpdaterType.GRU} or, {MemoryUpdaterType.RNN}"
        )


def get_learning_type(learning_type: str) -> LearningType:
    if LearningType(learning_type) is LearningType.SelfSupervised:
        return LearningType.SelfSupervised

    elif LearningType(learning_type) is LearningType.Supervised:
        return LearningType.Supervised

    else:
        raise Exception(
            f"Wrong learning type, expected {LearningType.Supervised} or, {LearningType.SelfSupervised}"
        )


def get_device_type(device_type: str) -> DeviceType:
    if DeviceType(device_type) is DeviceType.CUDA:
        return DeviceType.CUDA

    elif DeviceType(device_type) is DeviceType.CPU:
        return DeviceType.CPU

    else:
        raise Exception(
            f"Wrong device type, expected {DeviceType.CUDA} or, {DeviceType.CPU}"
        )
