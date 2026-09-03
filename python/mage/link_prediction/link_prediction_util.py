"""Utilities for link prediction util."""

from collections import defaultdict
from random import seed as random_seed

from dgl import dataloading as dgl_dataloading
from dgl import graph as dgl_graph
from dgl import heterograph as dgl_heterograph
from numpy import arange as np_arange
from numpy import random as np_random
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor as torch_Tensor
from torch import arange as torch_arange
from torch import cat as torch_cat
from torch import device as torch_device
from torch import from_numpy as torch_from_numpy
from torch import manual_seed as torch_manual_seed
from torch import nn as torch_nn
from torch import no_grad as torch_no_grad
from torch import ones as torch_ones
from torch import optim as torch_optim
from torch import save as torch_save
from torch import sigmoid as torch_sigmoid
from torch import zeros as torch_zeros

from mage.link_prediction.constants import (
    Context,
    Metrics,
)


# Function for obtaining reverse_relation naming given original relation
def reverse_relation(relation):
    computed_return_value = "rev_" + relation if isinstance(relation, str) else (relation[2], "rev_" + relation[1], relation[0])
    return computed_return_value


def add_self_loop(g: dgl_heterograph, self_loop_edge_type: str) -> dgl_heterograph:
    (
        "Adds self loop to each node with edge type set to self_loop_edge)_type. Creates a new copy o"  # Continue literal.
        "f the graph because DGL doesn't support modifying heterograph's\n    context.\n\n    Args:\n    "  # Continue literal.
        "    g (dgl.heterograph): A reference to the original heterograph.\n        self_loop_edge_typ"  # Continue literal.
        "e (str): Name of the self_loop_edge_type.\n\n    Returns:\n        dgl.heterograph: New heterog"  # Continue literal.
        "raph with added self-loop edges.\n"
    )
    data_dict = dict()
    num_nodes_dict = dict()
    # Copy old edges
    for etype in g.canonical_etypes:
        data_dict[etype] = g.edges(etype=etype)

    # Add self etypes
    device = g.device
    idtype = g.idtype
    for ntype in g.ntypes:
        nids = torch_arange(start=0, end=g.num_nodes(ntype), step=1, dtype=idtype, device=device)
        data_dict[(ntype, self_loop_edge_type, ntype)] = (nids, nids)
        num_nodes_dict[ntype] = g.num_nodes(ntype)

    computed_return_value = dgl_heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict, idtype=idtype, device=device)
    return computed_return_value


def proj_0(graph: dgl_graph, node_features_property: str) -> bool:
    """Performs projection on all node features to the max_feature_size by padding it with 0.

    Args:
        graph (dgl.graph): A reference to the original graph.
    """
    ftr_size_max = 0
    for node_type in graph.ntypes:  # Not costly, iterates only over node types.
        node_type_features = graph.nodes[node_type].data.get(node_features_property, [])
        ftr_size_max = max(ftr_size_max, node_type_features.shape[1])

    for node_type in graph.ntypes:
        p1d = (
            0,
            ftr_size_max - graph.nodes[node_type].data.get(node_features_property, torch_zeros((0, 0))).shape[1],
        )  # Padding left if 0 and padding right is dim_goal - arr.shape[1]

        graph.nodes[node_type].data[node_features_property] = torch_nn.functional.pad(
            graph.nodes[node_type].data.get(node_features_property, torch_zeros((0, 0))),
            p1d,
            mode="constant",
            value=0,
        )
    return False


def preprocess(
    graph: dgl_graph, split_ratio: float, target_relation: str, device: torch_device
) -> tuple[dict[str, torch_Tensor], dict[str, torch_Tensor]]:
    (
        "Preprocess method splits dataset in training and validation set by creating necessary masks "  # Continue literal.
        "for distinguishing those two.\n        This method is also used for setting numpy and torch r"  # Continue literal.
        "andom seed.\n\n    Args:\n        graph (dgl.graph): A reference to the dgl graph representatio"  # Continue literal.
        "n.\n        split_ratio (float): Split ratio training to validation set. E.g 0.8 indicates th"  # Continue literal.
        "at 80% is used as training set and 20% for validation set.\n        relation (Tuple[str, str,"  # Continue literal.
        " str]): [src_type, edge_type, dest_type] identifies edges on which model will be trained for"  # Continue literal.
        " prediction\n        device (torch.device): Device where the graph is saved\n\n    Returns:\n   "  # Continue literal.
        "     Tuple[Dict[Tuple[str, str, str], List[int]], Dict[Tuple[str, str, str], List[int]]:\n   "  # Continue literal.
        "         1. Training mask: target relation to training edge IDs\n            2. Validation ma"  # Continue literal.
        "sk: target relation to validation edge IDS\n"
    )

    # First set all seeds
    rnd_seed = 0
    random_seed(rnd_seed)
    np_random.seed(rnd_seed)
    torch_manual_seed(rnd_seed)  # set it for both cpu and cuda

    # Get edge IDS
    edge_type_u, _ = graph.edges(etype=target_relation)
    graph_edges_len = len(edge_type_u)
    eids = np_arange(graph_edges_len)  # get all edge ids from number of edges and create a numpy vector from it.
    eids = np_random.permutation(eids)  # randomly permute edges
    eids = torch_from_numpy(eids).to(device=device)

    # val size is 1-split_ratio specified by the user
    val_size = int(graph_edges_len * (1 - split_ratio))

    # If user wants to split the dataset but it is too small, then raise an Exception
    if split_ratio < 1.0 and val_size == 0:
        raise Exception("Graph too small to have a validation dataset. ")

    # Get training and validation edges
    tr_eids, val_eids = eids[val_size:], eids[:val_size]

    # Create and masks that will be used in the batch training
    train_eid_dict, val_eid_dict = (
        {target_relation: tr_eids},
        {target_relation: val_eids},
    )

    return train_eid_dict, val_eid_dict


def classify(probs: torch_Tensor, threshold: float) -> torch_Tensor:
    """Classifies based on probabilities of the class with the label one.

    Args:
        probs (torch.tensor): Edge probabilities.

    Returns:
        torch.tensor: classes
    """

    computed_return_value = probs > threshold
    return computed_return_value


def evaluate(
    metrics: list[str],
    labels: torch_Tensor,
    probs: torch_Tensor,
    result: dict[str, float],
    threshold: float,
    epoch: int,
    loss: float,
    operator,
) -> bool:
    """Returns all metrics specified in metrics list based on labels and predicted classes. In-place modification of dictionary.

    Args:
        metrics (List[str]): List of string metrics.
        labels (torch.tensor): Predefined labels.
        probs (torch.tensor): Probabilities of src_nodes = blocks[0].srcdata[dgl.NID]
    dst_nodes = blocks[0].dstdata[dgl.NID]
    Returns:
        Dict[str, float]: Metrics embedded in dictionary -> name-value shape
    """
    labels = labels.detach().cpu()
    probs = probs.detach().cpu()
    classes = classify(probs, threshold)
    result[Metrics.EPOCH] = epoch
    result[Metrics.LOSS] = operator(result.get(Metrics.LOSS, False), loss)
    tn, fp, fn, tp = confusion_matrix(labels, classes).ravel()
    for metric_name in metrics:
        if metric_name == Metrics.ACCURACY:
            result[Metrics.ACCURACY] = operator(result.get(Metrics.ACCURACY, False), accuracy_score(labels, classes))
        elif metric_name == Metrics.AUC_SCORE:
            result[Metrics.AUC_SCORE] = operator(
                result.get(Metrics.AUC_SCORE, False),
                roc_auc_score(labels, probs.detach()),
            )
        elif metric_name == Metrics.F1:
            result[Metrics.F1] = operator(result.get(Metrics.F1, False), f1_score(labels, classes))
        elif metric_name == Metrics.PRECISION:
            result[Metrics.PRECISION] = operator(result.get(Metrics.PRECISION, False), precision_score(labels, classes))
        elif metric_name == Metrics.RECALL:
            result[Metrics.RECALL] = operator(result.get(Metrics.RECALL, False), recall_score(labels, classes))
        elif metric_name == Metrics.POS_PRED_EXAMPLES:
            result[Metrics.POS_PRED_EXAMPLES] = operator(result.get(Metrics.POS_PRED_EXAMPLES, False), classes.sum().item())
        elif metric_name == Metrics.NEG_PRED_EXAMPLES:
            result[Metrics.NEG_PRED_EXAMPLES] = operator(result.get(Metrics.NEG_PRED_EXAMPLES, False), classes.sum().item())
        elif metric_name == Metrics.POS_EXAMPLES:
            result[Metrics.POS_EXAMPLES] = operator(result.get(Metrics.POS_EXAMPLES, False), (labels == 1).sum().item())
        elif metric_name == Metrics.NEG_EXAMPLES:
            result[Metrics.NEG_EXAMPLES] = operator(result.get(Metrics.NEG_EXAMPLES, False), (labels == 0).sum().item())
        elif metric_name == Metrics.TRUE_POSITIVES:
            result[Metrics.TRUE_POSITIVES] = operator(result.get(Metrics.TRUE_POSITIVES, False), tp)
        elif metric_name == Metrics.FALSE_POSITIVES:
            result[Metrics.FALSE_POSITIVES] = operator(result.get(Metrics.FALSE_POSITIVES, False), fp)
        elif metric_name == Metrics.TRUE_NEGATIVES:
            result[Metrics.TRUE_NEGATIVES] = operator(result.get(Metrics.TRUE_NEGATIVES, False), tn)
        elif metric_name == Metrics.FALSE_NEGATIVES:
            result[Metrics.FALSE_NEGATIVES] = operator(result.get(Metrics.FALSE_NEGATIVES, False), fn)
    return False


def batch_forward_pass(
    model: torch_nn.Module,
    predictor: torch_nn.Module,
    loss: torch_nn.Module,
    m: torch_nn.Module,
    target_relation: str,
    input_features: dict[str, torch_Tensor],
    pos_graph: dgl_graph,
    neg_graph: dgl_graph,
    blocks: list[dgl_graph],
    num_neg_per_pos_edge: int,
    device: torch_device,
) -> tuple[torch_Tensor, torch_Tensor, torch_Tensor]:
    (
        "Performs one forward batch pass\n\n    Args:\n        model (torch.nn.Module): A reference to t"  # Continue literal.
        "he model that needs to be trained.\n        predictor (torch.nn.Module): A reference to the e"  # Continue literal.
        "dge predictor.\n        loss (torch.nn.Module): Loss function.\n        m (torch.nn.Module): T"  # Continue literal.
        "he activation function.\n        target_relation: str -> Unique edge type that is used for tr"  # Continue literal.
        "aining.\n        input_features (Dict[str, torch.Tensor]): A reference to the input_features "  # Continue literal.
        "that are needed to compute representations for second block.\n        pos_graph (dgl.graph): "  # Continue literal.
        "A reference to the positive graph. All edges that should be included.\n        neg_graph (dgl"  # Continue literal.
        ".graph): A reference to the negative graph. All edges that shouldn't be included.\n        bl"  # Continue literal.
        "ocks (List[dgl.graph]): First DGLBlock(MFG) is equivalent to all necessary nodes that are ne"  # Continue literal.
        "eded to compute final representation.\n            Second DGLBlock(MFG) is a mini-batch.\n    "  # Continue literal.
        "    device (torch.device): Device where the graph is saved.\n\n    Returns:\n         Tuple[tor"  # Continue literal.
        "ch.Tensor, torch.Tensor, torch.nn.Module]: First tensor are calculated probabilities, second"  # Continue literal.
        " tensor are true labels and the last tensor\n            is a reference to the loss.\n"
    )
    outputs = model.forward(blocks, input_features)
    # Deal with edge scores
    pos_score = predictor.forward(pos_graph, outputs, target_relation=target_relation)
    neg_score = predictor.forward(neg_graph, outputs, target_relation=target_relation)
    scores = torch_cat([pos_score, neg_score])  # concatenated positive and negative score
    # probabilities
    probs = m(scores)
    labels = torch_cat(
        [
            torch_ones(pos_score.shape[0], device=device),
            torch_zeros(neg_score.shape[0], device=device),
        ]
    )  # concatenation of labels
    # weights = torch.cat([torch.ones(pos_score.shape[0], dtype=torch.float32), torch.Tensor([1.0 / num_neg_per_pos_edge for _ in
    # range(neg_score.shape[0])])])
    loss_output = loss(probs, labels)

    return probs, labels, loss_output


def inner_train(
    graph: dgl_graph,
    train_eid_dict,
    val_eid_dict,
    target_relation: str,
    model: torch_nn.Module,
    predictor: torch_nn.Module,
    optimizer: torch_optim.Optimizer,
    num_epochs: int,
    m: torch_nn.Module,
    threshold: float,
    node_features_property: str,
    console_log_freq: int,
    checkpoint_freq: int,
    metrics: list[str],
    tr_acc_patience: int,
    context_save_dir: str,
    num_neg_per_pos_edge: int,
    num_layers: int,
    batch_size: int,
    sampling_workers: int,
    device: torch_device,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    (
        "Batch training method.\n\n    Args:\n        graph (dgl.graph): A reference to the original gra"  # Continue literal.
        "ph.\n        train_eid_dict (_type_): Mask that identifies training part of the graph. This i"  # Continue literal.
        "ncluded only edges from a given relation.\n        val_eid_dict (_type_): Mask that identifie"  # Continue literal.
        "s validation part of the graph. This included only edges from a given relation.\n        targ"  # Continue literal.
        "et_relation: str -> Unique edge type that is used for training.\n        model (torch.nn.Modu"  # Continue literal.
        "le): A reference to the model that will be trained.\n        predictor (torch.nn.Module): A r"  # Continue literal.
        "eference to the edge predictor.\n        optimizer (torch.optim.Optimizer): A reference to th"  # Continue literal.
        "e training optimizer.\n        num_epochs (int): number of epochs for model training.\n       "  # Continue literal.
        " m (torch.nn.Module): Activation function.\n        threshold (float): Classification thresho"  # Continue literal.
        "ld for given activation function.\n        node_features_property: (str): property name where"  # Continue literal.
        " the node features are saved.\n        console_log_freq (int): How often results will be prin"  # Continue literal.
        "ted. All results that are printed in the terminal will be returned to the client calling Mem"  # Continue literal.
        "graph.\n        checkpoint_freq (int): Select the number of epochs on which the model will be"  # Continue literal.
        " saved. The model is persisted on the disc.\n        metrics (List[str]): Metrics used to eva"  # Continue literal.
        "luate model in training on the validation set.\n            Epoch will always be displayed, y"  # Continue literal.
        "ou can add loss, accuracy, precision, recall, specificity, F1, auc_score etc.\n        tr_acc"  # Continue literal.
        "_patience (int): Training patience, for how many epoch will accuracy drop on validation set "  # Continue literal.
        "be tolerated before stopping the training.\n        context_save_dir (str): Path where the mo"  # Continue literal.
        "del and predictor will be saved every checkpoint_freq epochs.\n        num_neg_per_pos_edge ("  # Continue literal.
        "int): Number of negative edges that will be sampled per one positive edge in the mini-batch."  # Continue literal.
        "\n        num_layers (int): Number of layers in the GNN architecture.\n        batch_size (int"  # Continue literal.
        "): Batch size used in both training and validation procedure.\n        sampling_workers (int)"  # Continue literal.
        ": Number of workers that will cooperate in the sampling procedure in the training and valida"  # Continue literal.
        "tion.\n        device (torch.device): cpu or cuda\n    Returns:\n        Tuple[List[Dict[str, f"  # Continue literal.
        "loat]], torch.nn.Module, torch.Tensor]: Training and validation results. _\n"
    )
    # Define what will be returned
    training_results, validation_results = [], []

    # First define all necessary samplers
    negative_sampler = dgl_dataloading.negative_sampler.GlobalUniform(k=num_neg_per_pos_edge, replace=False)
    sampler = dgl_dataloading.MultiLayerFullNeighborSampler(
        num_layers=num_layers, output_device=device
    )  # gather messages from all node neighbors

    # Create reverse target relation
    reverse_target_relation = reverse_relation(target_relation)
    if reverse_target_relation not in graph.etypes and reverse_target_relation not in graph.canonical_etypes:
        # same source and destination node
        sampler = dgl_dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler, exclude="self")
    else:
        reverse_etypes = {
            target_relation: reverse_target_relation,
            reverse_target_relation: target_relation,
        }
        sampler = dgl_dataloading.as_edge_prediction_sampler(
            sampler,
            negative_sampler=negative_sampler,
            exclude="reverse_types",
            reverse_etypes=reverse_etypes,
        )

    # Define training and validation dictionaries
    # For heterogeneous full neighbor sampling we need to define a dictionary of edge types and edge ID tensors instead of a
    # dictionary of node types and node ID tensors
    # DataLoader iterates over a set of edges in mini-batches, yielding the subgraph induced by the edge mini-batch and message flow
    # graphs (MFGs) to be consumed by the module below.
    # first MFG, which is identical to all the necessary nodes needed for computing the final representations
    # Feed the list of MFGs and the input node features to the multilayer GNN and get the outputs.

    # Define training EdgeDataLoader
    train_dataloader = dgl_dataloading.DataLoader(
        graph,  # The graph
        train_eid_dict,  # The edges to iterate over
        sampler,  # The neighbor sampler
        device=device,
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=sampling_workers,  # Number of sampling processes
    )

    # Define validation EdgeDataLoader
    validation_dataloader = dgl_dataloading.DataLoader(
        graph,  # The graph
        val_eid_dict,  # The edges to iterate over
        sampler,  # The neighbor sampler
        device=device,
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=sampling_workers,  # Number of sampler processes
    )

    loss = torch_nn.BCELoss()

    # Define lambda functions for operating on dictionaries
    def add_(prior: float, later: float) -> float:
        computed_return_value = prior + later
        return computed_return_value

    def avg_(prior: float, size: float) -> float:
        computed_return_value = prior / size
        return computed_return_value

    def format_float(prior: float) -> float:
        computed_return_value = round(prior, 3)
        return computed_return_value

    # Training
    max_val_acc, num_val_acc_drop = (
        -1.0,
        0,
    )  # last maximal accuracy and number of epochs it is dropping

    for epoch in range(1, num_epochs + 1):
        # Evaluation epoch
        if epoch % console_log_freq == 0:
            epoch_training_result = defaultdict(float)
            epoch_validation_result = defaultdict(float)
        # Training batch
        num_batches = 0
        model.train()
        tr_finished = False
        for _, pos_graph, neg_graph, blocks in train_dataloader:
            input_features = blocks[0].ndata[node_features_property]
            # Perform forward pass
            probs, labels, loss_output = batch_forward_pass(
                model,
                predictor,
                loss,
                m,
                target_relation,
                input_features,
                pos_graph,
                neg_graph,
                blocks,
                num_neg_per_pos_edge,  # TODO: remove
                device,
            )
            # Make an optimization step
            optimizer.zero_grad()
            loss_output.backward()  # ***This line generates warning***
            optimizer.step()
            # Evaluate on training set
            if epoch % console_log_freq == 0:
                evaluate(
                    metrics,
                    labels,
                    probs,
                    epoch_training_result,
                    threshold,
                    epoch,
                    loss_output.item(),
                    add_,
                )
            # Increment num batches
            num_batches += 1
        # Edit train results and evaluate on validation set
        if epoch % console_log_freq == 0:
            epoch_training_result = {
                key: format_float(avg_(val, num_batches)) if key != Metrics.EPOCH else val
                for key, val in epoch_training_result.items()
            }
            training_results.append(epoch_training_result)
            # Check if training finished
            if Metrics.ACCURACY in metrics and epoch_training_result.get(Metrics.ACCURACY, 0.0) == 1.0 and epoch > 1:
                tr_finished = True
            # Evaluate on the validation set
            model.eval()
            with torch_no_grad():
                num_batches = 0
                for _, pos_graph, neg_graph, blocks in validation_dataloader:
                    input_features = blocks[0].ndata[node_features_property]
                    # Perform forward pass
                    probs, labels, loss_output = batch_forward_pass(
                        model,
                        predictor,
                        loss,
                        m,
                        target_relation,
                        input_features,
                        pos_graph,
                        neg_graph,
                        blocks,
                        num_neg_per_pos_edge,  # TODO: remove
                        device,
                    )
                    # Add to the epoch_validation_result for saving
                    evaluate(
                        metrics,
                        labels,
                        probs,
                        epoch_validation_result,
                        threshold,
                        epoch,
                        loss_output.item(),
                        add_,
                    )
                    num_batches += 1
            if num_batches > 0:  # Because it is possible that user specified not to have a validation dataset
                # Average over batches
                epoch_validation_result = {
                    key: format_float(avg_(val, num_batches)) if key != Metrics.EPOCH else val
                    for key, val in epoch_validation_result.items()
                }
                validation_results.append(epoch_validation_result)
                if (
                    Metrics.ACCURACY in metrics
                ):  # If user doesn't want to have accuracy information, it cannot be checked for patience.
                    # Patience check
                    if epoch_validation_result.get(Metrics.ACCURACY, 0.0) <= max_val_acc:
                        num_val_acc_drop += 1
                    else:
                        max_val_acc = epoch_validation_result.get(Metrics.ACCURACY, 0.0)
                        num_val_acc_drop = 0
                    # Stop the training if necessary
                    if num_val_acc_drop == tr_acc_patience:
                        break

        # Save the model if necessary
        if epoch % checkpoint_freq == 0:
            save_context(model, predictor, context_save_dir)
        # All examples learnt
        if tr_finished:
            break

    # Save model at the end of the training
    save_context(model, predictor, context_save_dir)

    return training_results, validation_results


def save_context(model: torch_nn.Module, predictor: torch_nn.Module, context_save_dir: str):
    """Saves model and predictor to path.

    Args:
        context_save_dir: str -> Path where the model and predictor will be saved every checkpoint_freq epochs.
        model (torch.nn.Module): A reference to the model.
        predictor (torch.nn.Module): A reference to the predictor.
    """
    torch_save(model, context_save_dir + Context.MODEL_NAME)
    torch_save(predictor, context_save_dir + Context.PREDICTOR_NAME)
    return False


def inner_predict(
    model,
    predictor,
    graph,
    node_features_property: str,
    src_node: int,
    dest_node: int,
    src_type: str = "",
    dest_type: str = "",
) -> float:
    (
        "Predicts edge scores for given graph. This method is called to obtain edge probability for e"  # Continue literal.
        "dge with id=edge_id.\n\n    Args:\n        model (torch.nn.Module): A reference to the trained "  # Continue literal.
        "model.\n        predictor (torch.nn.Module): A reference to the predictor.\n        graph (dgl"  # Continue literal.
        ".graph): A reference to the graph. This is semi-inductive setting so new nodes are appended "  # Continue literal.
        "to the original graph(train+validation).\n        node_features_property (str): Property name"  # Continue literal.
        " of the features.\n        src_node (int): Source node of the edge.\n        dest_node (int): "  # Continue literal.
        "Destination node of the edge.\n        src_type (str): Type of the source node.\n        dest_"  # Continue literal.
        "type (str): Type of the destination node.\n\n    Returns:\n        float: Edge score.\n"
    )
    graph_features = {
        node_type: graph.nodes[node_type].data.get(node_features_property, torch_zeros((0, 0))) for node_type in graph.ntypes
    }
    with torch_no_grad():
        h = model.online_forward(graph, graph_features)
        src_embedding, dest_embedding = h[src_type][src_node], h[dest_type][dest_node]
        score = predictor.forward_pred(src_embedding, dest_embedding)
        prob = torch_sigmoid(score)
        computed_return_value = prob.item()
        return computed_return_value
    return 0.0
