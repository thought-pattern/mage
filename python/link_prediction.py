"""Utilities for link prediction."""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field, fields
from heapq import heappop, heappush
from itertools import chain as itertools_chain
from json import loads as js_loads
from math import isfinite

from dgl import AddReverse
from dgl import graph as dgl_graph  # geometric deep learning
from dgl import heterograph as dgl_heterograph
from dgl import remove_edges as dgl_remove_edges
from mgp import Any as mgp_Any  # Python API
from mgp import Label as mgp_Label
from mgp import List as mgp_List
from mgp import Map as mgp_Map
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch import cuda as torch_cuda
from torch import device as torch_device
from torch import equal as torch_equal
from torch import float32 as torch_float32
from torch import isin as torch_isin
from torch import load as torch_load
from torch import nn as torch_nn
from torch import optim as torch_optim
from torch import tensor as torch_tensor

from mage.link_prediction import (
    Activations,
    Aggregators,
    Context,
    Devices,
    Metrics,
    Models,
    Optimizers,
    Parameters,
    Predictors,
    Reindex,
    add_self_loop,
    classify,
    inner_train,
    preprocess,
    proj_0,
)
from mage.link_prediction.link_prediction_util import compute_node_embeddings, reverse_relation, score_pair
from mage.link_prediction.models.gat import GAT
from mage.link_prediction.models.graph_sage import GraphSAGE
from mage.link_prediction.predictors.DotPredictor import DotPredictor
from mage.link_prediction.predictors.MLPPredictor import MLPPredictor

##############################
# classes and data structures
##############################


@dataclass
class LinkPredictionParameters:
    (
        "Parameters user in LinkPrediction module.\n    :param in_feats: int -> Defines the size of th"  # Continue literal.
        "e input features. It will be automatically inferred by algorithm.\n    :param hidden_features"  # Continue literal.
        "_size: List[int] -> Defines the size of each hidden layer in the architecture.\n    :param la"  # Continue literal.
        "yer_type: str -> Layer type\n    :param num_epochs: int -> Number of epochs for model trainin"  # Continue literal.
        "g\n    :param optimizer: str -> Can be one of the following: ADAM, SGD, AdaGrad...\n    :param"  # Continue literal.
        " learning_rate: float -> Learning rate for optimizer\n    :param split_ratio: float -> Split "  # Continue literal.
        "ratio between training and validation set. There is not test dataset because it is assumed t"  # Continue literal.
        "hat user first needs to create new edges in dataset to test a model on them.\n    :param node"  # Continue literal.
        "_features_property: str → Property name where the node features are saved.\n    :param device"  # Continue literal.
        "_type: str ->  If model will be trained using CPU or cuda GPU. Possible values are cpu and c"  # Continue literal.
        "uda. To run it on Cuda, user must set this flag to true and system must support cuda executi"  # Continue literal.
        "on.\n        System's support is checked with torch.cuda.is_available()\n    :param console_lo"  # Continue literal.
        "g_freq: int ->  how often do you want to print results from your model? Results will be from"  # Continue literal.
        " validation dataset.\n    :param checkpoint_freq: int → Select the number of epochs on which "  # Continue literal.
        "the model will be saved. The model is persisted on disc.\n    :param aggregator: str → Aggreg"  # Continue literal.
        "ator used in models. Can be one of the following: lstm, pool, mean, gcn. It is only used in "  # Continue literal.
        "graph_sage, not graph_attn\n    :param metrics: mgp.List[str] -> Metrics used to evaluate mod"  # Continue literal.
        "el in training on the test/validation set(we don't use validation set to optimize parameters"  # Continue literal.
        " so everything is test set).\n        Epoch will always be displayed, you can add loss, accur"  # Continue literal.
        "acy, precision, recall, specificity, F1, auc_score etc.\n    :param predictor_type: str -> Ty"  # Continue literal.
        "pe of the predictor. Predictor is used for combining node scores to edge scores.\n    :param "  # Continue literal.
        "attn_num_heads: List[int] -> GAT can support usage of more than one head in each layers exce"  # Continue literal.
        "pt last one. Only used in GAT, not in GraphSage.\n    :param tr_acc_patience: int -> Training"  # Continue literal.
        " patience, for how many epoch will accuracy drop on validation set be tolerated before stopp"  # Continue literal.
        "ing the training.\n    :param context_save_dir: str -> Path where the model and predictor wil"  # Continue literal.
        "l be saved every checkpoint_freq epochs.\n    :param target_relation: str -> Unique edge type"  # Continue literal.
        " that is used for training.\n    :param num_neg_per_pos_edge (int): Number of negative edges "  # Continue literal.
        "that will be sampled per one positive edge in the mini-batch.\n    :param num_layers (int): N"  # Continue literal.
        "umber of layers in the GNN architecture.\n    :param batch_size (int): Batch size used in bot"  # Continue literal.
        "h training and validation procedure.\n    :param sampling_workers (int): Number of workers th"  # Continue literal.
        "at will cooperate in the sampling procedure in the training and validation.\n    :param last_"  # Continue literal.
        "activation_function (str) → Activation function that is applied after the last layer in the "  # Continue literal.
        "model and before the predictor_type. Currently, only sigmoid is supported.\n    :param add_re"  # Continue literal.
        "verse_edges (bool) -> Whether the module should add reverse edges for each in the obtained g"  # Continue literal.
        "raph. If the source and destination nodes are of the same type, edges of the same edge type "  # Continue literal.
        "will\n            be created. If the source and destination nodes are different, then prefix "  # Continue literal.
        "rev_ will be added to the previous edge type. Reverse edges will be excluded as message pass"  # Continue literal.
        "ing edges for corresponding supervision edges.\n    :param add_self_loops (bool) -> Whether t"  # Continue literal.
        'he module should add self loop edges to every node in the graph with edge_type set to "self"'  # Continue literal.
        ".\n\n"
    )

    in_feats: int = 0
    hidden_features_size: list = field(
        default_factory=lambda: [128, 128]
        # Cannot add typing because of the way Python is implemented(no default things in dataclass, list is immutable something
        # like
        # this)
    )
    layer_type: str = Models.GRAPH_ATTN
    num_epochs: int = 10
    optimizer: str = Optimizers.ADAM_OPT
    learning_rate: float = 0.01
    split_ratio: float = 0.8
    node_features_property: str = "features"
    device_type: str = Devices.CPU_DEVICE
    console_log_freq: int = 1
    checkpoint_freq: int = 10
    aggregator: str = Aggregators.POOL_AGG
    metrics: list = field(
        default_factory=lambda: [
            Metrics.LOSS,
            Metrics.ACCURACY,
            Metrics.AUC_SCORE,
            Metrics.PRECISION,
            Metrics.RECALL,
            Metrics.F1,
            Metrics.TRUE_POSITIVES,
            Metrics.FALSE_POSITIVES,
            Metrics.TRUE_NEGATIVES,
            Metrics.FALSE_NEGATIVES,
        ]
    )
    predictor_type: str = Predictors.MLP_PREDICTOR
    attn_num_heads: list[int] = field(default_factory=lambda: [4, 4])
    tr_acc_patience: int = 5
    context_save_dir: str = "/tmp/"
    target_relation: str = ""
    num_neg_per_pos_edge: int = 1
    batch_size: int = 512
    sampling_workers: int = 4
    last_activation_function = Activations.SIGMOID
    add_reverse_edges = False  # only allowed in some cases
    add_self_loops = False  # for automatically adding self-loop


##############################
# global parameters
##############################

link_prediction_parameters: LinkPredictionParameters = LinkPredictionParameters()  # parameters currently saved.
# Names set_model_parameters accepts: the dataclass fields plus the three unannotated class-level switches.
SETTABLE_PARAMETERS = {parameter.name for parameter in fields(LinkPredictionParameters)} | {
    Parameters.LAST_ACTIVATION_FUNCTION.value,
    Parameters.ADD_REVERSE_EDGES.value,
    Parameters.ADD_SELF_LOOPS.value,
}
training_results: list[dict[str, float]] = (
    list()
)  # List of all output training records. String is the metric's name and float represents value.
validation_results: list[dict[str, float]] = (
    list()
)  # List of all output validation results. String is the metric's name and float represents value in the Dictionary inside.
graph: mgp_Any = False
reindex: mgp_Any = {}  # Mapping of DGL indexes to original dataset indexes for all node types and reverse.
predictor: mgp_Any = False  # Predictor for calculating edge scores
model: mgp_Any = False
# Resolved architecture and graph contract of the published model and predictor (the checkpoint manifest); empty until
# train or load_model publishes all three together.
trained_manifest: dict = {}
labels_concat = ":"  # string to separate labels if dealing with multiple labels per node
# Device where the model is executed; set_model_parameters selects CUDA when configured and available.
device: mgp_Any = torch_device(Devices.CPU_DEVICE)

feat_drop_rate = 0.09164
attn_drop_rate = 0.09164
alpha_rate = 0.512857
res_def = True


# Lambda function to concat list of labels
def merge_labels(labels: list[mgp_Label]) -> str:
    computed_return_value = labels_concat.join([label.name for label in labels])
    return computed_return_value


##############################
# All read procedures
##############################


@mgp_read_proc
def set_model_parameters(ctx: mgp_ProcCtx, parameters: mgp_Map) -> mgp_Record:
    (
        "Saves parameters to the global parameters link_prediction_parameters. Specific parsing is ne"  # Continue literal.
        "eded because we want enable user to call it with a subset of parameters, no need to send the"  # Continue literal.
        "m all.\n    We will use some kind of reflection to most easily update parameters.\n\n    Args:\n"  # Continue literal.
        "        ctx (mgp.ProcCtx):  Reference to the context execution.\n        hidden_features_size"  # Continue literal.
        ": mgp.List[int] -> Defines the size of each hidden layer in the architecture.\n        layer_"  # Continue literal.
        "type: str -> Layer type\n        num_epochs: int -> Number of epochs for model training\n     "  # Continue literal.
        "   optimizer: str -> Can be one of the following: ADAM, SGD, AdaGrad...\n        learning_rat"  # Continue literal.
        "e: float -> Learning rate for optimizer\n        split_ratio: float -> Split ratio between tr"  # Continue literal.
        "aining and validation set. There is not test dataset because it is assumed that user first n"  # Continue literal.
        "eeds to create new edges in dataset to test a model on them.\n        node_features_property:"  # Continue literal.
        " str → Property name where the node features are saved.\n        device_type: str ->  If mode"  # Continue literal.
        "l will be trained using CPU or cuda GPU. Possible values are cpu and cuda. To run it on Cuda"  # Continue literal.
        ", user must set this flag to true and system must support cuda execution.\n                  "  # Continue literal.
        "              System's support is checked with torch.cuda.is_available()\n        console_log"  # Continue literal.
        "_freq: int ->  how often do you want to print results from your model? Results will be from "  # Continue literal.
        "validation dataset.\n        checkpoint_freq: int → Select the number of epochs on which the "  # Continue literal.
        "model will be saved. The model is persisted on disc.\n        aggregator: str → Aggregator us"  # Continue literal.
        "ed in models. Can be one of the following: lstm, pool, mean, gcn.\n        metrics: mgp.List["  # Continue literal.
        "str] -> Metrics used to evaluate model in training.\n        predictor_type str: Type of the "  # Continue literal.
        "predictor. Predictor is used for combining node scores to edge scores.\n        attn_num_head"  # Continue literal.
        "s: List[int] -> GAT can support usage of more than one head in each layer except last one. O"  # Continue literal.
        "nly used in GAT, not in GraphSage.\n        tr_acc_patience: int -> Training patience, for ho"  # Continue literal.
        "w many epoch will accuracy drop on test set be tolerated before stopping the training.\n     "  # Continue literal.
        "   context_save_dir: str -> Path where the model and predictor will be saved every checkpoin"  # Continue literal.
        "t_freq epochs.\n        target_relation: str -> Unique edge type that is used for training.\n "  # Continue literal.
        "       num_neg_per_pos_edge: int -> Number of negative edges that will be sampled per one po"  # Continue literal.
        "sitive edge in the mini-batch.\n        batch_size : Batch size used in both training and val"  # Continue literal.
        "idation procedure.\n        sampling_workers (int): Number of workers that will cooperate in "  # Continue literal.
        "the sampling procedure in the training and validation.\n        last_activation_function (str"  # Continue literal.
        ") → Activation function that is applied after the last layer in the model and before the pre"  # Continue literal.
        "dictor_type. Currently, only sigmoid is supported.\n        add_reverse_edges (bool) -> Wheth"  # Continue literal.
        "er the module should add reverse edges for each in the obtained graph. If the source and des"  # Continue literal.
        "tination nodes are of the same type, edges of the same edge type will\n            be created"  # Continue literal.
        ". If the source and destination nodes are different, then prefix rev_ will be added to the p"  # Continue literal.
        "revious edge type. Reverse edges will be excluded as message passing edges for corresponding"  # Continue literal.
        " supervision edges.\n        add_self_loops (bool) -> Whether the module should add self loop"  # Continue literal.
        ' edges to every node in the graph with edge_type set to "self".\n\n    Returns:\n        mgp.Re'  # Continue literal.
        "cord:\n            status (bool): True if parameters were successfully updated, False otherwi"  # Continue literal.
        "se.\n            message(str): Additional explanation why method failed or OK otherwise.\n"
    )
    global link_prediction_parameters, device

    requested = dict(parameters)
    unknown = sorted(key for key in requested if key not in SETTABLE_PARAMETERS)
    if unknown:
        computed_return_value = mgp_Record(status=False, message=f"Unknown parameter(s): {', '.join(unknown)}. ")
        return computed_return_value

    validate_user_parameters(parameters=requested)

    # The update is applied to a copy and published only after the complete effective configuration is admitted, so a
    # rejected request leaves the prior configuration and device untouched.
    candidate = copy(link_prediction_parameters)
    for key, value in requested.items():
        setattr(candidate, key, value)
    for key in (Parameters.HIDDEN_FEATURES_SIZE, Parameters.ATTN_NUM_HEADS, Parameters.METRICS):
        value = getattr(candidate, key)
        if isinstance(value, tuple):
            setattr(candidate, key, list(value))

    validate_effective_parameters(candidate)

    candidate_device = (
        torch_device(Devices.CUDA_DEVICE)
        if candidate.device_type == Devices.CUDA_DEVICE and torch_cuda.is_available()
        else torch_device(Devices.CPU_DEVICE)
    )
    if candidate_device.type == "cuda":
        candidate.sampling_workers = 0

    link_prediction_parameters, device = candidate, candidate_device

    computed_return_value = mgp_Record(status=True, message="OK")
    return computed_return_value


@mgp_read_proc
def train(
    ctx: mgp_ProcCtx,
) -> mgp_Record:
    (
        "Train method is used for training the module on the dataset provided with ctx. By taking dec"  # Continue literal.
        "ision to split the dataset here and not in the separate method, it is impossible to retrain "  # Continue literal.
        "the same model.\n\n    Args:\n        ctx (mgp.ProcCtx, optional): Reference to the process exe"  # Continue literal.
        "cution.\n\n    Returns:\n        mgp.Record: It returns performance metrics obtained during the"  # Continue literal.
        " training on the training and validation dataset.\n"
    )
    # Get global context
    global training_results, validation_results, predictor, model, graph, reindex, trained_manifest

    # Reset parameters of the old training
    reset_train_predict_parameters()

    parameters = link_prediction_parameters
    graph_contract = {
        "node_features_property": parameters.node_features_property,
        "target_relation": parameters.target_relation,
        "add_reverse_edges": parameters.add_reverse_edges,
        "add_self_loops": parameters.add_self_loops,
    }

    # Get some data
    # Dealing with heterogeneous graphs
    graph, reindex, type_triplets = get_dgl_graph_data(ctx, graph_contract)

    # An omitted target relation ("") is inferred only when the graph has exactly one edge type.
    target_relation = parameters.target_relation
    if not target_relation:
        if len(type_triplets) != 1:
            raise ValueError("target_relation must be set when the graph has more than one edge type. ")
        target_relation = type_triplets[0]

    # An omitted input width (0) is the converted feature width; an explicit one must match it.
    feature_width = node_feature_width(graph, parameters.node_features_property)
    in_feats = parameters.in_feats if parameters.in_feats else feature_width
    if in_feats != feature_width:
        raise ValueError(f"in_feats is {in_feats} but the node features are {feature_width} wide. ")

    # Split the data
    train_eid_dict, val_eid_dict = preprocess(
        graph=graph,
        split_ratio=parameters.split_ratio,
        target_relation=target_relation,
        device=device,
    )
    if len(train_eid_dict.get(target_relation, [])) == 0:
        raise ValueError(f"split_ratio {parameters.split_ratio} leaves no training edges. ")

    # Extract number of layers
    num_layers = len(parameters.hidden_features_size)

    # Plain values only (no enum members), so the manifest loads under torch.load(weights_only=True).
    checkpoint_manifest = {
        "format": Context.CHECKPOINT_FORMAT.value,
        "layer_type": parameters.layer_type.lower(),
        "in_feats": in_feats,
        "hidden_features_size": list(parameters.hidden_features_size),
        "attn_num_heads": list(parameters.attn_num_heads),
        "aggregator": parameters.aggregator.lower(),
        "predictor_type": parameters.predictor_type.lower(),
        "edge_types": list(graph.etypes),
        "target_relation": target_relation,
        "node_features_property": parameters.node_features_property,
        "add_reverse_edges": parameters.add_reverse_edges,
        "add_self_loops": parameters.add_self_loops,
        "last_activation_function": parameters.last_activation_function.lower(),
    }
    trained_model, trained_predictor = construct_architecture(checkpoint_manifest)

    optimizer_types: dict[mgp_Any, mgp_Any] = {
        Optimizers.ADAM_OPT: torch_optim.Adam,
        Optimizers.SGD_OPT: torch_optim.SGD,
    }
    optimizer_type = parameters.optimizer.upper()
    optimizer_class = optimizer_types.get(optimizer_type, torch_optim.Optimizer)
    if optimizer_class is torch_optim.Optimizer:
        raise ValueError(f"Optimizer {optimizer_type} is not supported")
    optimizer = optimizer_class(
        itertools_chain(trained_model.parameters(), trained_predictor.parameters()),
        lr=parameters.learning_rate,
    )

    activation_function = parameters.last_activation_function
    if activation_function != Activations.SIGMOID:
        raise ValueError(f"Activation function {activation_function} is not supported")
    m = torch_nn.Sigmoid()
    threshold = 0.5

    # Call training method
    new_training_results, new_validation_results = inner_train(
        graph,
        train_eid_dict,
        val_eid_dict,
        target_relation,
        trained_model,
        trained_predictor,
        optimizer,
        parameters.num_epochs,
        m,
        threshold,
        parameters.node_features_property,
        parameters.console_log_freq,
        parameters.checkpoint_freq,
        parameters.metrics,
        parameters.tr_acc_patience,
        parameters.context_save_dir,
        parameters.num_neg_per_pos_edge,
        num_layers,
        parameters.batch_size,
        parameters.sampling_workers,
        device,
        checkpoint_manifest,
    )

    # Publish the trained model, its predictor and their contract together, only after training completed.
    training_results, validation_results = new_training_results, new_validation_results
    model, predictor, trained_manifest = trained_model, trained_predictor, checkpoint_manifest

    # Return results
    computed_return_value = mgp_Record(training_results=training_results, validation_results=validation_results)
    return computed_return_value


@mgp_read_proc
def predict(ctx: mgp_ProcCtx, src_vertex: mgp_Vertex, dest_vertex: mgp_Vertex) -> mgp_Record:
    (
        "Predict method. It is assumed that nodes are added to the original Memgraph graph. It suppor"  # Continue literal.
        "ts both situations, when the edge doesn't exist and when\n    the edge exists.\n\n    Args:\n   "  # Continue literal.
        "     ctx (mgp.ProcCtx): A reference to the context execution\n        src_vertex (mgp.Vertex)"  # Continue literal.
        ": Source vertex.\n        dest_vertex (mgp.Vertex): Destination vertex.\n\n    Returns:\n       "  # Continue literal.
        " score: probability that two nodes are connected\n"
    )
    global graph, reindex

    # If the model isn't available. Model is available if this method is called right after training or loaded from context.
    # Same goes for predictor.
    if not trained_manifest or model is False or predictor is False:
        raise Exception("No trained model available to the system. Train or load it first. ")

    # Load graph again so you find nodes that were possibly added between train and prediction. It is built under the
    # trained contract (feature property, reverse edges, self loops), not whatever the configuration says now.
    graph, reindex, _ = get_dgl_graph_data(ctx, trained_manifest)
    admit_trained_feature_width(graph)
    target_relation = trained_manifest.get("target_relation", "")

    # Create dgl graph representation
    src_old_id, src_type = src_vertex.id, merge_labels(src_vertex.labels)
    dest_old_id, dest_type = dest_vertex.id, merge_labels(dest_vertex.labels)

    # Check if src_type and dest_type are of the same target relation
    if isinstance(target_relation, tuple):
        if src_type != target_relation[0] or dest_type != target_relation[2]:
            raise Exception("Prediction can be only computed on edges on which model was trained. ")
    else:
        for etype in graph.canonical_etypes:
            if target_relation == etype[1] and (etype[0] != src_type or etype[2] != dest_type):
                raise Exception("Prediction can be only computed on edges on which model was trained. ")

    # Get dgl ids
    src_id = reindex.get(Reindex.MEMGRAPH, {})[src_type][src_old_id]
    dest_id = reindex.get(Reindex.MEMGRAPH, {})[dest_type][dest_old_id]

    message_graph = mask_target_bindings(graph, target_relation, src_id, [dest_id])
    embeddings = compute_node_embeddings(model, message_graph, trained_manifest.get("node_features_property", ""))
    score = score_pair(predictor, embeddings, src_id, dest_id, src_type, dest_type)

    result = mgp_Record(score=score)
    return result


@mgp_read_proc
def recommend(
    ctx: mgp_ProcCtx,
    src_vertex: mgp_Vertex,
    dest_vertices: mgp_List[mgp_Vertex],
    k: int,
) -> list[mgp_Record]:
    (
        "Recommend method. It is assumed that nodes are already added to the original graph and our g"  # Continue literal.
        "oal is to predict whether there is an edge between two nodes or not. Even if the edge exists"  # Continue literal.
        ",\n     method can be used. Recommends k nodes based on edge scores.\n\n\n    Args:\n        ctx "  # Continue literal.
        "(mgp.ProcCtx): A reference to the context execution\n        src_vertex (mgp.Vertex): Source "  # Continue literal.
        "vertex.\n        dest_vertex (mgp.Vertex): Destination vertex.\n\n    Returns:\n        score: P"  # Continue literal.
        "robability that two nodes are connected\n"
    )
    global graph, reindex

    # If the model isn't available
    if not trained_manifest or model is False or predictor is False:
        raise Exception("No trained model available to the system. Train or load it first. ")

    # You called predict after session was lost
    graph, reindex, _ = get_dgl_graph_data(ctx, trained_manifest)
    admit_trained_feature_width(graph)
    target_relation = trained_manifest.get("target_relation", "")

    # Create dgl graph representation
    src_old_id, src_type = src_vertex.id, merge_labels(src_vertex.labels)

    # Check if src_type is of the same target relation
    if isinstance(target_relation, tuple):
        if src_type != target_relation[0]:
            raise Exception("Prediction can be only computed on edges on which model was trained. ")
    else:
        for etype in graph.canonical_etypes:
            if target_relation == etype[1] and etype[0] != src_type:
                raise Exception("Prediction can be only computed on edges on which model was trained. ")

    # Get dgl ids
    src_id = reindex.get(Reindex.MEMGRAPH, {})[src_type][src_old_id]

    # Admit every candidate before any scoring: (vertex, dgl id, type).
    candidates: list[tuple[mgp_Vertex, int, str]] = []
    for dest_vertex in dest_vertices:
        # Get dest vertex
        dest_old_id, dest_type = dest_vertex.id, merge_labels(dest_vertex.labels)
        dest_id = reindex.get(Reindex.MEMGRAPH, {})[dest_type][dest_old_id]

        # Check if dest_type is of the same target relation
        if isinstance(target_relation, tuple):
            if dest_type != target_relation[2]:
                raise Exception("Prediction can be only computed on edges on which model was trained. ")
        else:
            for etype in graph.canonical_etypes:
                if target_relation == etype[1] and etype[2] != dest_type:
                    raise Exception("Prediction can be only computed on edges on which model was trained. ")
        candidates.append((dest_vertex, dest_id, dest_type))

    if not candidates:
        computed_return_value = []
        return computed_return_value

    # Whether each candidate edge already exists; these are the "relevant" labels for the diagnostic metrics.
    dest_ids = [dest_id for _, dest_id, _ in candidates]
    existing_edges = graph.has_edges_between(
        torch_tensor([src_id] * len(dest_ids), dtype=graph.idtype, device=graph.device),
        torch_tensor(dest_ids, dtype=graph.idtype, device=graph.device),
        etype=target_relation,
    ).tolist()

    # One masked message graph and one embedding pass serve every candidate pair.
    message_graph = mask_target_bindings(graph, target_relation, src_id, dest_ids)
    embeddings = compute_node_embeddings(model, message_graph, trained_manifest.get("node_features_property", ""))

    # Save edge scores and vertices for each dest vertex.
    results: list[tuple[float, int, mgp_Vertex]] = []
    for i, (dest_vertex, dest_id, dest_type) in enumerate(candidates):
        score = score_pair(predictor, embeddings, src_id, dest_id, src_type, dest_type)
        heappush(results, (-score, i, dest_vertex))  # Build in O(n). Add i to break ties where all predict values are the same.

    # Extract recommendations and metrics
    top_recommendations, top_scores, top_labels = (
        [],
        [],
        [],
    )  # scores=probability, recommendation=mgp.Vertex, labels save info if edges exist or not
    pop_size = min(k, len(results))

    if trained_manifest.get("last_activation_function", Activations.SIGMOID) == Activations.SIGMOID:
        threshold = 0.5
    else:
        raise Exception(f"Currently, only {Activations.SIGMOID} is supported. ")

    for _ in range(pop_size):
        score, candidate_index, recommendation = heappop(results)
        if -score < threshold:  # No need to continue because that is not predicted edge
            break
        # Handle vertex
        top_recommendations.append(recommendation)  # vertices
        # Handle score
        top_scores.append(-score)  # floats
        # Handle labels
        top_labels.append(1 if existing_edges[candidate_index] else 0)

    # Update k value
    new_k = len(top_scores)

    # Calculate recommendation metrics. Recommending absent (novel) links is the procedure's purpose, so ratios that are
    # undefined for an empty selection or a selection without existing edges report 0 instead of aborting the results.
    precision_at_k, recall_at_k = 0.0, 0.0
    if top_scores:
        top_classes = classify(torch_tensor(top_scores), threshold)
        precision_at_k = precision_score(top_labels, top_classes, zero_division=0.0)
        recall_at_k = recall_score(top_labels, top_classes, zero_division=0.0)
    f1_denominator = precision_at_k + recall_at_k
    f1_at_k = 2 * precision_at_k * recall_at_k / f1_denominator if f1_denominator > 0 else 0.0
    ap_text = "undefined (no existing edge among the recommendations)"
    if any(top_labels):
        ap_text = str(round(float(average_precision_score(top_labels, top_scores)), 3))  # average precision

    # Create final return results
    return_results = []
    for i in range(len(top_scores)):
        return_results.append(mgp_Record(score=top_scores[i], recommendation=top_recommendations[i]))

    print("*** Recommendation metrics ***")
    print(f"Precision@{new_k}: {round(float(precision_at_k), 3)}")
    print(f"Recall@{new_k}: {round(float(recall_at_k), 3)} ")
    print(f"F1@{new_k}: {round(float(f1_at_k), 3)}")
    print(f"AP: {ap_text}")

    return return_results


@mgp_read_proc
def get_training_results(
    ctx: mgp_ProcCtx,
) -> mgp_Record:
    (
        "This method is used when user wants to get performance data obtained from the last training."  # Continue literal.
        " It is in the form of list of records where each record is a Dict[metric_name, metric_value]"  # Continue literal.
        ". Training and validation\n    results are returned.\n\n    Args:\n        ctx (mgp.ProcCtx): Re"  # Continue literal.
        "ference to the context execution\n\n    Returns:\n        mgp.Record[List[LinkPredictionOutputR"  # Continue literal.
        "esult]]: A list of results. If the train method wasn't called yet, it returns empty lists.\n"
    )
    global training_results, validation_results

    if not training_results or not validation_results:
        raise Exception("Training results are outdated or train method wasn't called. ")

    computed_return_value = mgp_Record(training_results=training_results, validation_results=validation_results)
    return computed_return_value


@mgp_read_proc
def load_model(ctx: mgp_ProcCtx, path: str = link_prediction_parameters.context_save_dir) -> mgp_Record:
    """Loads the checkpoint bundle train saved under the given directory. If the path doesn't exist, underlying exception
    is thrown. If the path argument is not given, it loads from the default path. If the user has changed path and the
    context was deleted then he/she needs to send that parameter here.

    The bundle is read with torch.load(weights_only=True): it holds tensor state dictionaries and a plain-value manifest,
    never pickled code. The model and predictor are rebuilt from the manifest and both state dictionaries must load
    exactly before the model, predictor and manifest are published together.

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution.

    Returns:
        status(mgp.Any): True just to indicate that loading went well.
    """

    global model, predictor, trained_manifest
    bundle = torch_load(path + Context.CHECKPOINT_NAME, map_location=device, weights_only=True)
    manifest = admit_checkpoint_manifest(bundle)
    loaded_model, loaded_predictor = construct_architecture(manifest)
    loaded_model.load_state_dict(bundle.get("model", {}))
    loaded_predictor.load_state_dict(bundle.get("predictor", {}))
    model, predictor, trained_manifest = loaded_model, loaded_predictor, manifest
    computed_return_value = mgp_Record(status=True)
    return computed_return_value


@mgp_read_proc
def reset_parameters(ctx: mgp_ProcCtx) -> mgp_Record:
    """Resets all parameters.

    Args:
        ctx (mgp.ProcCtx): A reference to the execution context.

    Returns:
        status: True if all passed ok.
    """
    reset_train_predict_parameters()
    computed_return_value = mgp_Record(status=True)
    return computed_return_value


##############################
# Private helper methods.
##############################
def construct_architecture(manifest: dict) -> tuple[torch_nn.Module, torch_nn.Module]:
    """Builds the model and predictor a checkpoint manifest describes; train and load_model share this owner."""
    hidden_features_size = manifest.get("hidden_features_size", [])
    num_layers = len(hidden_features_size)
    layer_type = manifest.get("layer_type", "")
    if layer_type == Models.GRAPH_SAGE:
        architecture_model = GraphSAGE(
            in_feats=manifest.get("in_feats", 0),
            hidden_features_size=hidden_features_size,
            aggregator=manifest.get("aggregator", ""),
            feat_drops=[feat_drop_rate for _ in range(num_layers)],
            edge_types=manifest.get("edge_types", []),
            device=device,
        )
    elif layer_type == Models.GRAPH_ATTN:
        architecture_model = GAT(
            in_feats=manifest.get("in_feats", 0),
            hidden_features_size=hidden_features_size,
            attn_num_heads=manifest.get("attn_num_heads", []),
            feat_drops=[feat_drop_rate for _ in range(num_layers)],
            attn_drops=[attn_drop_rate for _ in range(num_layers)],
            alphas=[alpha_rate for _ in range(num_layers)],
            residuals=[res_def for _ in range(num_layers)],
            edge_types=manifest.get("edge_types", []),
            device=device,
        )
    else:
        raise ValueError(f"Layer type {layer_type} is not supported")

    predictor_type = manifest.get("predictor_type", "")
    if predictor_type == Predictors.DOT_PREDICTOR:
        architecture_predictor = DotPredictor()
    elif predictor_type == Predictors.MLP_PREDICTOR:
        architecture_predictor = MLPPredictor(h_feats=hidden_features_size[-1], device=device)
    else:
        raise ValueError(f"Predictor type {predictor_type} is not supported")
    architecture = (architecture_model, architecture_predictor)
    return architecture


def admit_checkpoint_manifest(bundle: object) -> dict:
    """Validates a loaded checkpoint bundle's layout and returns its manifest."""
    if not isinstance(bundle, dict):
        raise ValueError("The link prediction checkpoint is not a checkpoint bundle. ")
    manifest = bundle.get("manifest", {})
    if not isinstance(manifest, dict) or manifest.get("format", "") != Context.CHECKPOINT_FORMAT:
        raise ValueError(f"The link prediction checkpoint is not in the {Context.CHECKPOINT_FORMAT.value} format. ")
    if not isinstance(bundle.get("model", False), dict) or not isinstance(bundle.get("predictor", False), dict):
        raise ValueError("The link prediction checkpoint lacks model or predictor state. ")
    manifest_types = {
        "layer_type": str,
        "in_feats": int,
        "hidden_features_size": list,
        "attn_num_heads": list,
        "aggregator": str,
        "predictor_type": str,
        "edge_types": list,
        "target_relation": (str, tuple),
        "node_features_property": str,
        "add_reverse_edges": bool,
        "add_self_loops": bool,
        "last_activation_function": str,
    }
    for key, expected_type in manifest_types.items():
        if key not in manifest or not isinstance(manifest.get(key, ""), expected_type):
            raise ValueError(f"The link prediction checkpoint manifest has no valid {key!r}. ")
    return manifest


def node_feature_width(graph: dgl_graph, node_features_property: str) -> int:
    """Feature width of the converted graph; proj_0 has padded every node type to the same width."""
    width = max(
        graph.nodes[node_type].data.get(node_features_property, torch_tensor([])).shape[1] for node_type in graph.ntypes
    )
    return width


def admit_trained_feature_width(prediction_graph: dgl_graph) -> bool:
    """Refuses a prediction graph whose feature width differs from the width the published model was trained on."""
    trained_width = trained_manifest.get("in_feats", 0)
    width = node_feature_width(prediction_graph, trained_manifest.get("node_features_property", ""))
    if width != trained_width:
        raise ValueError(f"The model was trained on {trained_width}-wide node features but the graph has {width}. ")
    return False


def mask_target_bindings(prediction_graph: dgl_graph, target_relation: object, src_id: int, dest_ids: list[int]) -> dgl_graph:
    """
    Message graph for scoring src -> dest candidate pairs. Training's edge sampler excludes the supervised edge (and its
    reverse-type counterpart) from message passing, so every existing target edge between a candidate pair is removed
    here the same way; a candidate pair is never inserted as an observation. The input graph is left unchanged.
    """
    candidates = torch_tensor(dest_ids, dtype=prediction_graph.idtype, device=prediction_graph.device)
    edge_src, edge_dest = prediction_graph.edges(etype=target_relation)
    target_eids = ((edge_src == src_id) & torch_isin(edge_dest, candidates)).nonzero().flatten()
    message_graph = dgl_remove_edges(prediction_graph, target_eids, etype=target_relation)

    reverse_target_relation = reverse_relation(target_relation)
    if reverse_target_relation in prediction_graph.etypes or reverse_target_relation in prediction_graph.canonical_etypes:
        reverse_src, reverse_dest = message_graph.edges(etype=reverse_target_relation)
        reverse_eids = (torch_isin(reverse_src, candidates) & (reverse_dest == src_id)).nonzero().flatten()
        message_graph = dgl_remove_edges(message_graph, reverse_eids, etype=reverse_target_relation)
    return message_graph


def admit_node_features(features: object, vertex_id: int, node_features_property: str) -> list[float]:
    """
    Node features are a list of finite numbers, or a string holding a JSON array of finite numbers. A data property is
    parsed as data and never evaluated as code.
    """
    values = features
    if isinstance(features, str):
        try:
            values = js_loads(features)
        except (ValueError, RecursionError) as err:
            raise ValueError(f"Vertex {vertex_id} property {node_features_property!r} is not a JSON array of numbers. ") from err
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"Vertex {vertex_id} property {node_features_property!r} must be a non-empty list of numbers. ")
    admitted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"Vertex {vertex_id} property {node_features_property!r} holds a non-numeric or non-finite value. ")
        admitted.append(float(value))
    return admitted


def process_help_function(
    mem_indexes: dict[str, int],
    old_index: int,
    type_: str,
    features: object,
    node_features_property: str,
    reindex: dict[str, dict[str, dict[int, int]]],
    index_dgl_to_features: dict[str, dict[int, list[float]]],
) -> bool:
    """Helper function for mapping original Memgraph graph to DGL representation.

    Args:
        mem_indexes (Dict[str, int]): Saves counters for each node type.
        old_index (int): Memgraph's node index.
        type_ (str): Node type.
        features (object): Node features property value.
        node_features_property (str): Property name where the node features are saved.
        reindex (Dict[str, Dict[str, Dict[int, int]]]): Mapping from original indexes to DGL indexes for each node type and reverse.
        index_dgl_to_features (Dict[str, Dict[int, List[float]]]): DGL indexes to features for each node type.
    """
    # Install each node type's maps before writing into them.
    dgl_to_memgraph = reindex.setdefault(Reindex.DGL, {}).setdefault(type_, {})
    memgraph_to_dgl = reindex.setdefault(Reindex.MEMGRAPH, {}).setdefault(type_, {})
    type_features = index_dgl_to_features.setdefault(type_, {})

    # Check if old_index has been seen for this label
    if old_index not in memgraph_to_dgl:
        ind = mem_indexes.get(type_, 0)  # get current counter
        type_features[ind] = admit_node_features(features, old_index, node_features_property)
        dgl_to_memgraph[ind] = old_index  # save new_to_old relationship
        memgraph_to_dgl[old_index] = ind  # save old_to_new relationship
        mem_indexes[type_] = ind + 1
    return False


def get_dgl_graph_data(
    ctx: mgp_ProcCtx,
    graph_contract: dict,
) -> tuple[dgl_graph, mgp_Any, list[tuple[str, str, str]]]:
    (
        "Creates dgl representation of the graph. It works with heterogeneous and homogeneous.\n\n    A"  # Continue literal.
        "rgs:\n        ctx (mgp.ProcCtx): The reference to the context execution.\n        graph_contract"  # Continue literal.
        " (dict): node_features_property, target_relation, add_reverse_edges and add_self_loops: the "  # Continue literal.
        "configuration for training, the trained manifest for prediction.\n\n    Returns:\n      "  # Continue literal.
        "  Tuple[dgl.graph, Dict[str, Dict[int32, int32]], List[Tuple[str, str, str]]]: DGL graph "  # Continue literal.
        "representation, mappings between DGL and Memgraph indexes for each node type in both\n    "  # Continue literal.
        "    directions, and the graph's own (src_type, edge_type, dst_type) triplets.\n"
    )

    node_features_property = graph_contract.get("node_features_property", "")
    target_relation = graph_contract.get("target_relation", "")

    reindex = {Reindex.DGL: {}, Reindex.MEMGRAPH: {}}  # map of label to new node index to old node index and reverse
    mem_indexes = {}  # map of label to the next DGL index of that label

    # list of tuples where each tuple is in following form(src_type, edge_type, dst_type), e.g. ("Customer", "SUBSCRIBES_TO",
    # "Plan")
    type_triplets = []
    index_dgl_to_features = {}  # dgl indexes to features

    src_nodes, dest_nodes = (
        defaultdict(list),
        defaultdict(list),
    )  # label to node IDs -> Tuple of node-tensors format from DGL

    edge_types = set()

    isolated_nodes = []  # saves old ids

    # Iterate over all vertices
    for vertex in ctx.graph.vertices:
        # Process source vertex
        src_id, src_type, src_features = (
            vertex.id,
            merge_labels(vertex.labels),
            vertex.properties.get(node_features_property, False),
        )

        # Find if the node is disconnected from the rest of the graph
        src_isolated_node = not (any(True for _ in vertex.in_edges) or any(True for _ in vertex.out_edges))

        # If it isn't isolated node than map all indexes. Must be done before iterating over outgoing edges.
        if not src_isolated_node:
            process_help_function(
                mem_indexes,
                src_id,
                src_type,
                src_features,
                node_features_property,
                reindex,
                index_dgl_to_features,
            )

        # Get all out edges
        for edge in vertex.out_edges:
            # Get edge information
            edge_type = edge.type.name

            # Process destination vertex next
            dest_node = edge.to_vertex
            dest_id, dest_type, dest_features = (
                dest_node.id,
                merge_labels(dest_node.labels),
                dest_node.properties.get(node_features_property, False),
            )

            # Define type triplet
            type_triplet = (src_type, edge_type, dest_type)
            # If this type triplet was already processed

            type_triplet_in = type_triplet in type_triplets

            # Before processing node dest node and edge, check if this edge_type has occurred with different src_type or dest_type
            if edge_type in edge_types and not type_triplet_in and edge_type == target_relation:
                raise Exception(
                    f"Edges of edge type {edge_type}"
                    f" are used for training and there are already edges with this edge type but with different co"
                    f"mbination of source and destination node. "
                )

            # Add to the type triplets
            if not type_triplet_in:
                type_triplets.append(type_triplet)

            # Add to the edge_types set
            edge_types.add(edge_type)

            # Handle mappings
            process_help_function(
                mem_indexes,
                dest_id,
                dest_type,
                dest_features,
                node_features_property,
                reindex,
                index_dgl_to_features,
            )

            # Define edge
            src_nodes[type_triplet].append(reindex.get(Reindex.MEMGRAPH, {})[src_type][src_id])
            dest_nodes[type_triplet].append(reindex.get(Reindex.MEMGRAPH, {})[dest_type][dest_id])

        # Append old id
        if src_isolated_node:
            isolated_nodes.append(src_id)

    # Check if there are no edges in the dataset, assume that it cannot learn effectively without edges. E2E handling.
    if len(src_nodes.keys()) == 0:
        raise Exception("No edges in the dataset. ")

    # data_dict has specific type that DGL requires to create a heterograph
    data_dict = dict()

    # Create a heterograph
    for type_triplet in type_triplets:
        data_dict[type_triplet] = (
            torch_tensor(src_nodes[type_triplet], device=device),
            torch_tensor(dest_nodes[type_triplet], device=device),
        )

    g = dgl_heterograph(data_dict, device=device)

    # Process isolated nodes by appending them to the end
    for isolated_node_id in isolated_nodes:
        isolated_node = ctx.graph.get_vertex_by_id(isolated_node_id)
        isolated_node_type, isolated_node_features = (
            merge_labels(isolated_node.labels),
            isolated_node.properties.get(node_features_property, False),
        )
        process_help_function(
            mem_indexes,
            isolated_node_id,
            isolated_node_type,
            isolated_node_features,
            node_features_property,
            reindex,
            index_dgl_to_features,
        )
        g.add_nodes(1, ntype=isolated_node_type)

    # Add undirected support if specified by user
    if graph_contract.get("add_reverse_edges", False):
        reverse_edges_transform = AddReverse(copy_edata=True, sym_new_etype=False)
        g = reverse_edges_transform(g)  # unfortunately copying is done

    # Custom made self-loop function is specified by the user.
    if graph_contract.get("add_self_loops", False):
        g = add_self_loop(g, "self")

    # Create features
    for node_type in g.ntypes:
        node_features = []
        for node in g.nodes(node_type):
            node_id = node.item()
            node_features.append(index_dgl_to_features.get(node_type, {})[node_id])

        if len({len(features) for features in node_features}) > 1:
            raise ValueError(f"Nodes of type {node_type} have {node_features_property!r} lists of different lengths. ")
        node_features = torch_tensor(node_features, dtype=torch_float32, device=device)
        g.nodes[node_type].data[node_features_property] = node_features

    # Test conversion. Note: Do a conversion before you upscale features.
    conversion_to_dgl_test(
        graph=g,
        reindex=reindex,
        ctx=ctx,
        node_features_property=node_features_property,
    )

    # Upscale features so they are all of same size
    proj_0(g, node_features_property)

    converted = (g, reindex, type_triplets)
    return converted


def reset_train_predict_parameters() -> bool:
    """Reset global parameters that are returned by train method and used by predict method."""
    global training_results, validation_results, predictor, model, graph, reindex, trained_manifest
    training_results = []  # clear training records from previous training
    validation_results = []  # clear validation record from previous training
    predictor = False  # Delete old predictor and create a new one in link_prediction_util.train method\
    model = False  # Annulate old model
    trained_manifest = {}  # The published model's contract goes with it
    graph = False  # Set graph to None
    reindex = False  # Delete indexing stuff
    return False


def conversion_to_dgl_test(
    graph: dgl_graph,
    reindex: dict[str, dict[str, dict[int, int]]],
    ctx: mgp_ProcCtx,
    node_features_property: str,
) -> bool:
    (
        "\n    Tests whether conversion from ctx.ProcCtx graph to dgl graph went successfully. Checks "  # Continue literal.
        "how features are mapped. Throws exception if something fails.\n\n    Args:\n        graph (dgl."  # Continue literal.
        "graph): Reference to the dgl heterogeneous graph.\n        reindex (Dict[str, Dict[str, Dict["  # Continue literal.
        "int, int]]]): Mapping from new indexes to old indexes for all node types and reverse.\n      "  # Continue literal.
        "  ctx (mgp.ProcCtx): Reference to the context execution.\n        node_features_property (str"  # Continue literal.
        "): Property namer where the node features are saved`\n"
    )

    # Check if the dataset is empty. E2E handling.
    if len(ctx.graph.vertices) == 0:
        raise Exception("The conversion to DGL failed. The dataset is empty. ")

    # Find all node types.
    for node_type in graph.ntypes:
        for vertex in graph.nodes(node_type):
            # Get int from torch.Tensor
            vertex_id = vertex.item()
            # Find vertex in Memgraph
            old_id = reindex.get(Reindex.DGL, {})[node_type][vertex_id]
            vertex = ctx.graph.get_vertex_by_id(old_id)
            if vertex is None:
                raise Exception(f"The conversion to DGL failed. Vertex with id {old_id} is not mapped to DGL graph. ")

            # Admit features exactly as the conversion did.
            old_features = admit_node_features(vertex.properties.get(node_features_property, False), old_id, node_features_property)

            # Check if equal
            if not torch_equal(
                graph.nodes[node_type].data.get(node_features_property, [])[vertex_id],
                torch_tensor(old_features, dtype=torch_float32, device=device),
            ):
                raise Exception(
                    "The conversion to DGL failed. Stored graph does not contain the same features as the converted DGL graph. "
                )
    return False


def validate_user_parameters(parameters: mgp_Map) -> bool:
    """Validates parameters user sent through method set_model_parameters

    Args:
        parameters (mgp.Map): Parameters sent by user.

    Returns:
        Nothing or raises an exception if something is wrong.
    """

    def type_checker(arg, message, real_type):
        """Raise when an argument does not have the required runtime type (a Boolean is not an int here)."""
        if not isinstance(arg, real_type) or (real_type is int and isinstance(arg, bool)):
            raise TypeError(message)
        return False

    # Input feature width: 0 infers it from the node features at training.
    if "in_feats" in parameters.keys():
        in_feats = parameters.get("in_feats", 0)
        type_checker(in_feats, "in_feats must be an int. ", int)
        if in_feats < 0:
            raise Exception("in_feats must be 0 (infer) or a positive feature width. ")

    # Hidden features size
    if Parameters.HIDDEN_FEATURES_SIZE in parameters.keys():
        hidden_features_size = parameters.get(Parameters.HIDDEN_FEATURES_SIZE, 0)

        # Because list cannot be sent through mgp.
        type_checker(hidden_features_size, "hidden_features_size not an iterable object. ", tuple)
        if not hidden_features_size:
            raise Exception("hidden_features_size must define at least one layer. ")

        for hid_size in hidden_features_size:
            type_checker(hid_size, "layer_size must be an int", int)
            if hid_size <= 0:
                raise Exception("Layer size must be greater than 0. ")

    # Layer type check
    if Parameters.LAYER_TYPE in parameters.keys():
        layer_type = parameters.get(Parameters.LAYER_TYPE, "")

        # Check typing
        type_checker(layer_type, "layer_type must be string. ", str)

        if layer_type != Models.GRAPH_ATTN and layer_type != Models.GRAPH_SAGE:
            raise Exception("Unknown layer type, this module supports only graph_attn and graph_sage. ")

    # Num epochs
    if Parameters.NUM_EPOCHS in parameters.keys():
        num_epochs = parameters.get(Parameters.NUM_EPOCHS, [])

        # Check typing
        type_checker(num_epochs, "num_epochs must be int. ", int)

        if num_epochs <= 0:
            raise Exception("Number of epochs must be greater than 0. ")

    # Optimizer check
    if Parameters.OPTIMIZER in parameters.keys():
        optimizer = parameters.get(Parameters.OPTIMIZER, False)

        # Check typing
        type_checker(optimizer, "optimizer must be a string. ", str)

        if optimizer != Optimizers.ADAM_OPT and optimizer != Optimizers.SGD_OPT:
            raise Exception("Unknown optimizer, this module supports only ADAM and SGD. ")

    # Learning rate check
    if Parameters.LEARNING_RATE in parameters.keys():
        learning_rate = parameters.get(Parameters.LEARNING_RATE, 0.0)

        # Check typing
        type_checker(learning_rate, "learning rate must be a float. ", float)

        if not isfinite(learning_rate) or learning_rate <= 0.0:
            raise Exception("Learning rate must be a finite number greater than 0. ")

    # Split ratio check
    if Parameters.SPLIT_RATIO in parameters.keys():
        split_ratio = parameters.get(Parameters.SPLIT_RATIO, False)

        # Check typing
        type_checker(split_ratio, "split_ratio must be a float. ", float)

        # 1.0 keeps every edge for training (no validation set); anything outside (0, 1] is not a split.
        if not isfinite(split_ratio) or split_ratio <= 0.0 or split_ratio > 1.0:
            raise Exception("Split ratio must be greater than 0 and at most 1. ")

    # node_features_property check
    if Parameters.NODE_FEATURES_PROPERTY in parameters.keys():
        node_features_property = parameters.get(Parameters.NODE_FEATURES_PROPERTY, False)

        # Check typing
        type_checker(node_features_property, "node_features_property must be a string. ", str)

        if node_features_property == "":
            raise Exception("You must specify name of nodes' features property. ")

    # device_type check
    if Parameters.DEVICE_TYPE in parameters.keys():
        device_type = parameters.get(Parameters.DEVICE_TYPE, "")

        # Check typing
        type_checker(device_type, "device_type must be a string. ", str)

        if device_type not in Devices:
            raise Exception("Only cpu and cuda are supported as devices. ")

    # console_log_freq check
    if Parameters.CONSOLE_LOG_FREQ in parameters.keys():
        console_log_freq = parameters.get(Parameters.CONSOLE_LOG_FREQ, False)

        # Check typing
        type_checker(console_log_freq, "console_log_freq must be an int. ", int)

        if console_log_freq <= 0:
            raise Exception("Console log frequency must be greater than 0. ")

    # checkpoint freq check
    if Parameters.CHECKPOINT_FREQ in parameters.keys():
        checkpoint_freq = parameters.get(Parameters.CHECKPOINT_FREQ, False)

        # Check typing
        type_checker(checkpoint_freq, "checkpoint_freq must be an int. ", int)

        if checkpoint_freq <= 0:
            raise Exception("Checkpoint frequency must be greater than 0. ")

    # aggregator check
    if Parameters.AGGREGATOR in parameters.keys():
        aggregator = parameters.get(Parameters.AGGREGATOR, False)

        # Check typing
        type_checker(aggregator, "aggregator must be a string. ", str)

        if aggregator not in Aggregators:
            raise Exception("Aggregator must be one of the following: mean, pool, lstm or gcn. ")

    # metrics check
    if Parameters.METRICS in parameters.keys():
        metrics = parameters.get(Parameters.METRICS, [])

        # Check typing
        type_checker(metrics, "metrics must be an iterable object. ", tuple)

        for metric in metrics:
            if metric.lower() not in Metrics:
                raise Exception("Metric name " + metric + " is not supported!")

    # Predictor type
    if Parameters.PREDICTOR_TYPE in parameters.keys():
        predictor_type = parameters.get(Parameters.PREDICTOR_TYPE, "")

        # Check typing
        type_checker(predictor_type, "predictor_type must be a string. ", str)

        if predictor_type not in Predictors:
            raise Exception("Predictor " + predictor_type + " is not supported. ")

    # Attention heads
    if Parameters.ATTN_NUM_HEADS in parameters.keys():
        attn_num_heads = parameters.get(Parameters.ATTN_NUM_HEADS, [])

        # Check typing
        type_checker(attn_num_heads, "attn_num_heads must be an iterable object. ", tuple)
        # The per-layer count is checked against the effective layer list in validate_effective_parameters.
        for num_heads in attn_num_heads:
            type_checker(num_heads, "attention head counts must be ints. ", int)
            if num_heads <= 0:
                raise Exception("GAT allows only positive, larger than 0 values for number of attention heads. ")

    # Training accuracy patience
    if Parameters.TR_ACC_PATIENCE in parameters.keys():
        tr_acc_patience = parameters.get(Parameters.TR_ACC_PATIENCE, False)

        # Check typing
        type_checker(tr_acc_patience, "tr_acc_patience must be an iterable object. ", int)

        if tr_acc_patience <= 0:
            raise Exception("Training acc patience flag must be larger than 0.")

    # model_save_path
    if Parameters.MODEL_SAVE_PATH in parameters.keys():
        model_save_path = parameters.get(Parameters.MODEL_SAVE_PATH, "")

        # Check typing
        type_checker(model_save_path, "model_save_path must be a string. ", str)

        if model_save_path == "":
            raise Exception("Path must be !=  ")

    # context save dir
    if Parameters.CONTEXT_SAVE_DIR in parameters.keys():
        context_save_dir = parameters.get(Parameters.CONTEXT_SAVE_DIR, {})

        # check typing
        type_checker(context_save_dir, "context_save_dir must be a string. ", str)

        if context_save_dir == "":
            raise Exception("Path must not be empty string. ")

    # target edge type
    if Parameters.TARGET_RELATION in parameters.keys():
        target_relation = parameters.get(Parameters.TARGET_RELATION, False)

        # check typing
        if not isinstance(target_relation, str) and not isinstance(target_relation, tuple):
            raise Exception("target relation must be a string or a tuple. ")
        if isinstance(target_relation, tuple) and (
            len(target_relation) != 3 or not all(isinstance(part, str) for part in target_relation)
        ):
            raise Exception("A target relation tuple must be [source type, edge type, destination type]. ")

    # num_neg_per_positive_edge
    if Parameters.NUM_NEG_PER_POS_EDGE in parameters.keys():
        num_neg_per_pos_edge = parameters.get(Parameters.NUM_NEG_PER_POS_EDGE, False)

        # Check typing
        type_checker(
            num_neg_per_pos_edge,
            "number of negative edges per positive one must be an int. ",
            int,
        )
        if num_neg_per_pos_edge <= 0:
            raise Exception("Number of negative edges per positive one must be greater than 0. ")

    # batch size
    if Parameters.BATCH_SIZE in parameters.keys():
        batch_size = parameters.get(Parameters.BATCH_SIZE, 0)

        # Check typing
        type_checker(batch_size, "batch_size must be an int", int)
        if batch_size <= 0:
            raise Exception("Batch size must be greater than 0. ")

    # sampling workers
    if Parameters.SAMPLING_WORKERS in parameters.keys():
        sampling_workers = parameters.get(Parameters.SAMPLING_WORKERS, [])

        # check typing
        type_checker(sampling_workers, "sampling_workers must be and int", int)
        if sampling_workers < 0:
            raise Exception("Number of sampling workers must not be negative. ")

    # last activation function
    if Parameters.LAST_ACTIVATION_FUNCTION in parameters.keys():
        last_activation_function = parameters.get(Parameters.LAST_ACTIVATION_FUNCTION, False)

        # check typing
        type_checker(last_activation_function, "last_activation_function should be a string", str)

        if last_activation_function != Activations.SIGMOID:
            raise Exception(f"Only {Activations.SIGMOID} is currently supported. ")

    # add reverse edges
    if Parameters.ADD_REVERSE_EDGES in parameters.keys():
        add_reverse_edges = parameters.get(Parameters.ADD_REVERSE_EDGES, [])

        # check typing
        type_checker(add_reverse_edges, "add_reverse_edges should be a bool. ", bool)

    # add_self_loops
    if Parameters.ADD_SELF_LOOPS in parameters.keys():
        add_self_loops = parameters.get(Parameters.ADD_SELF_LOOPS, [])

        # check typing
        type_checker(add_self_loops, "add_self_loops should be a bool. ", bool)
    return False


def validate_effective_parameters(parameters: LinkPredictionParameters) -> bool:
    """Checks cross-field constraints on the complete configuration an update would publish (current plus requested)."""
    if parameters.layer_type == Models.GRAPH_ATTN and len(parameters.attn_num_heads) != len(parameters.hidden_features_size):
        raise Exception(
            f"Specified network with {len(parameters.hidden_features_size)} layers but given attention heads data for "
            f"{len(parameters.attn_num_heads)} layers. "
        )
    return False


def get_number_of_edges(ctx: mgp_ProcCtx) -> int:
    """Returns number of edges for graph from execution context.

    Args:
        ctx (mgp.ProcCtx): A reference to the execution context.

    Returns:
        int: A number of edges.
    """
    edge_cnt = 0
    for vertex in ctx.graph.vertices:
        edge_cnt += len(list(vertex.out_edges))
    return edge_cnt
