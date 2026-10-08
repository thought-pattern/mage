"""Utilities for link prediction util."""

from functools import partial as functools_partial
from os import fdopen as os_fdopen, fsync as os_fsync, path as os_path, replace as os_replace, unlink as os_unlink
from random import seed as random_seed
from tempfile import mkstemp

from dgl import dataloading as dgl_dataloading, graph as dgl_graph, heterograph as dgl_heterograph
from numpy import arange as np_arange, random as np_random
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)
from torch import (
    Tensor as torch_Tensor,
    arange as torch_arange,
    cat as torch_cat,
    device as torch_device,
    from_numpy as torch_from_numpy,
    manual_seed as torch_manual_seed,
    nn as torch_nn,
    no_grad as torch_no_grad,
    ones as torch_ones,
    optim as torch_optim,
    save as torch_save,
    sigmoid as torch_sigmoid,
    tensor as torch_tensor,
    unique as torch_unique,
    zeros as torch_zeros,
)

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


def preprocess(graph: dgl_graph, split_ratio: float, target_relation, device: torch_device) -> tuple[dict, dict]:
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


def classify(probs, threshold: float):
    """Classifies based on probabilities of the class with the label one.

    Args:
        probs (torch.tensor or numpy.ndarray): Edge probabilities.

    Returns:
        Boolean classes of the same array type as probs.
    """

    computed_return_value = probs > threshold
    return computed_return_value


def confusion_counts(labels, classes) -> tuple[int, int, int, int]:
    """Returns (true negatives, false positives, false negatives, true positives) for binary labels and classes.

    Both binary classes are always counted, so an input holding a single class is counted rather than rejected.
    """
    tn, fp, fn, tp = (int(count) for count in confusion_matrix(labels, classes, labels=[0, 1]).ravel())
    counts = tn, fp, fn, tp
    return counts


def zero_safe_rate(numerator: int, denominator: int) -> float:
    """Returns one classification rate, or 0.0 when its denominator is zero and the rate is undefined."""
    rate = numerator / denominator if denominator else 0.0
    return rate


def accumulate_batch(statistics: dict, labels: torch_Tensor, probs: torch_Tensor, loss_value: float) -> bool:
    """Adds one batch to an epoch's sufficient statistics: its labels, its probabilities and its example-weighted
    loss. Metrics are derived once per epoch from these (epoch_metrics), so counts are totals and rates are global
    rather than means of per-batch values.

    Args:
        statistics (dict): The epoch's statistics; an empty dict starts a new epoch.
        labels (torch.tensor): True labels of the batch (1 positive, 0 negative).
        probs (torch.tensor): Predicted probabilities of the batch.
        loss_value (float): Mean loss of the batch.
    """
    batch_labels = statistics.get("labels", [])
    batch_labels.append(labels.detach().cpu())
    statistics["labels"] = batch_labels
    batch_probs = statistics.get("probs", [])
    batch_probs.append(probs.detach().cpu())
    statistics["probs"] = batch_probs
    statistics["loss_sum"] = statistics.get("loss_sum", 0.0) + loss_value * labels.shape[0]
    return False


def epoch_metrics(metrics: list[str], statistics: dict, threshold: float, epoch: int) -> dict:
    """Returns the epoch number, the example-weighted loss and every requested metric, derived from the whole epoch.

    The confusion matrix always has both binary classes, so batches or epochs with a single class (for example no
    sampled negatives) are counted rather than rejected. Precision, recall and F1 are 0.0 when their denominator is
    zero. AUC is undefined unless the epoch has both classes; it is then reported as NaN.

    Args:
        metrics (List[str]): Requested metric names.
        statistics (dict): Non-empty statistics built by accumulate_batch.
        threshold (float): Probability above which an example is classified positive.
        epoch (int): Epoch number.

    Returns:
        Dict[str, float]: Metric name to value; rates rounded to three decimals, counts exact.
    """
    labels = torch_cat(statistics.get("labels", [])).numpy().astype(int)
    probs = torch_cat(statistics.get("probs", [])).numpy()
    classes = classify(probs, threshold).astype(int)
    example_count = labels.shape[0]
    tn, fp, fn, tp = confusion_counts(labels, classes)
    positive_predictions = int(classes.sum())
    positive_examples = int(labels.sum())

    result = {Metrics.EPOCH: epoch, Metrics.LOSS: round(statistics.get("loss_sum", 0.0) / example_count, 3)}
    for metric_name in metrics:
        if metric_name == Metrics.ACCURACY:
            result[Metrics.ACCURACY] = round((tp + tn) / example_count, 3)
        elif metric_name == Metrics.AUC_SCORE:
            both_classes = 0 < positive_examples < example_count
            result[Metrics.AUC_SCORE] = round(float(roc_auc_score(labels, probs)), 3) if both_classes else float("nan")
        elif metric_name == Metrics.F1:
            result[Metrics.F1] = round(zero_safe_rate(2 * tp, 2 * tp + fp + fn), 3)
        elif metric_name == Metrics.PRECISION:
            result[Metrics.PRECISION] = round(zero_safe_rate(tp, tp + fp), 3)
        elif metric_name == Metrics.RECALL:
            result[Metrics.RECALL] = round(zero_safe_rate(tp, tp + fn), 3)
        elif metric_name == Metrics.POS_PRED_EXAMPLES:
            result[Metrics.POS_PRED_EXAMPLES] = positive_predictions
        elif metric_name == Metrics.NEG_PRED_EXAMPLES:
            result[Metrics.NEG_PRED_EXAMPLES] = example_count - positive_predictions
        elif metric_name == Metrics.POS_EXAMPLES:
            result[Metrics.POS_EXAMPLES] = positive_examples
        elif metric_name == Metrics.NEG_EXAMPLES:
            result[Metrics.NEG_EXAMPLES] = example_count - positive_examples
        elif metric_name == Metrics.TRUE_POSITIVES:
            result[Metrics.TRUE_POSITIVES] = tp
        elif metric_name == Metrics.FALSE_POSITIVES:
            result[Metrics.FALSE_POSITIVES] = fp
        elif metric_name == Metrics.TRUE_NEGATIVES:
            result[Metrics.TRUE_NEGATIVES] = tn
        elif metric_name == Metrics.FALSE_NEGATIVES:
            result[Metrics.FALSE_NEGATIVES] = fn
    return result


def exclude_held_out(target_etypes: list, held_out_ids: torch_Tensor, seed_edges: dict) -> dict:
    """DGL exclusion rule for one minibatch: the edges kept out of message passing. These are every held-out
    (validation) binding of the target relation plus the minibatch's own supervision edges, each together with its
    reverse edge, so neither the batch's targets nor validation topology inform the embeddings being trained or
    evaluated.

    Args:
        target_etypes (list): The canonical target relation, followed by its reverse relation when the graph has one;
            edge i of the reverse relation is the reverse of edge i of the target relation.
        held_out_ids (torch.Tensor): Target-relation edge IDs whose (source, destination) binding is held out.
        seed_edges (dict): The minibatch's supervision edges by canonical edge type.

    Returns:
        dict: Edge IDs to exclude by canonical edge type.
    """
    batch_ids = seed_edges.get(target_etypes[0], held_out_ids[:0])
    excluded_ids = torch_unique(torch_cat([held_out_ids.to(batch_ids.device), batch_ids]))
    excluded = {etype: excluded_ids for etype in target_etypes}
    return excluded


def batch_forward_pass(
    model: torch_nn.Module,
    predictor: torch_nn.Module,
    loss: torch_nn.Module,
    m: torch_nn.Module,
    target_relation,
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
    target_relation,
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
    checkpoint_manifest: dict,
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
        "tion.\n        device (torch.device): cpu or cuda\n        checkpoint_manifest (dict): Architecture"  # Continue literal.
        " and graph contract saved with every checkpoint so it can be rebuilt.\n    Returns:\n"  # Continue literal.
        "        Tuple[List[Dict[str, f"  # Continue literal.
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
    target_etypes = [graph.to_canonical_etype(target_relation)]
    if reverse_target_relation in graph.etypes or reverse_target_relation in graph.canonical_etypes:
        target_etypes.append(graph.to_canonical_etype(reverse_target_relation))

    # Held-out bindings: every target edge whose (source, destination) pair is a validation edge, parallel edges
    # included. They and their reverses stay out of the message graph for training and validation alike, while the
    # validation edge IDs remain the supervision set of the validation loader.
    edge_sources, edge_destinations = graph.edges(etype=target_etypes[0])
    validation_ids = val_eid_dict.get(target_relation, torch_zeros(0, dtype=edge_sources.dtype))
    held_out_bindings = set(zip(edge_sources[validation_ids].tolist(), edge_destinations[validation_ids].tolist()))
    held_out_ids = torch_tensor(
        [
            edge_id
            for edge_id, binding in enumerate(zip(edge_sources.tolist(), edge_destinations.tolist()))
            if binding in held_out_bindings
        ],
        dtype=edge_sources.dtype,
        device=edge_sources.device,
    )
    sampler = dgl_dataloading.as_edge_prediction_sampler(
        sampler,
        negative_sampler=negative_sampler,
        exclude=functools_partial(exclude_held_out, target_etypes, held_out_ids),
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

    # Training
    max_val_acc, num_val_acc_drop = (
        -1.0,
        0,
    )  # last maximal accuracy and number of epochs it is dropping

    for epoch in range(1, num_epochs + 1):
        # Evaluation epoch
        training_statistics = {}
        validation_statistics = {}
        # Training batch
        model.train()
        predictor.train()
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
                accumulate_batch(training_statistics, labels, probs, loss_output.item())
        # Edit train results and evaluate on validation set
        if epoch % console_log_freq == 0 and training_statistics:
            epoch_training_result = epoch_metrics(metrics, training_statistics, threshold, epoch)
            training_results.append(epoch_training_result)
            # Check if training finished
            if Metrics.ACCURACY in metrics and epoch_training_result.get(Metrics.ACCURACY, 0.0) == 1.0 and epoch > 1:
                tr_finished = True
            # Evaluate on the validation set
            model.eval()
            predictor.eval()
            with torch_no_grad():
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
                    # Add to the epoch's validation statistics
                    accumulate_batch(validation_statistics, labels, probs, loss_output.item())
            if validation_statistics:  # Because it is possible that user specified not to have a validation dataset
                epoch_validation_result = epoch_metrics(metrics, validation_statistics, threshold, epoch)
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
            save_context(model, predictor, checkpoint_manifest, context_save_dir)
        # All examples learnt
        if tr_finished:
            break

    # Save model at the end of the training
    save_context(model, predictor, checkpoint_manifest, context_save_dir)

    return training_results, validation_results


def save_context(model: torch_nn.Module, predictor: torch_nn.Module, checkpoint_manifest: dict, context_save_dir: str):
    """Publishes one checkpoint bundle: the model and predictor tensor state dictionaries plus the manifest needed to
    rebuild them. The bundle holds only tensors and plain values, so it loads under torch.load(weights_only=True). It is
    written beside its destination, flushed, then renamed over it, so readers see the previous or the new generation.

    Args:
        model (torch.nn.Module): A reference to the model.
        predictor (torch.nn.Module): A reference to the predictor.
        checkpoint_manifest (dict): Architecture and graph contract of model and predictor.
        context_save_dir: str -> Path where the checkpoint will be saved every checkpoint_freq epochs.
    """
    checkpoint_path = context_save_dir + Context.CHECKPOINT_NAME
    bundle = {
        "manifest": checkpoint_manifest,
        "model": model.state_dict(),
        "predictor": predictor.state_dict(),
    }
    staging_descriptor, staging_path = mkstemp(dir=os_path.dirname(checkpoint_path) or ".", suffix=".staging")
    try:
        with os_fdopen(staging_descriptor, "wb") as staging_file:
            torch_save(bundle, staging_file)
            staging_file.flush()
            os_fsync(staging_file.fileno())
        os_replace(staging_path, checkpoint_path)
    except BaseException:
        os_unlink(staging_path)
        raise
    return False


def compute_node_embeddings(model, graph, node_features_property: str) -> dict[str, torch_Tensor]:
    """Computes every node's embedding once on the given message graph; any number of pairs can then be scored.

    Args:
        model (torch.nn.Module): A reference to the trained model.
        graph (dgl.graph): Message graph. This is semi-inductive setting so new nodes are appended to the original graph.
        node_features_property (str): Property name of the features.

    Returns:
        Dict[str, torch.Tensor]: Embeddings for every node type.
    """
    graph_features = {
        node_type: graph.nodes[node_type].data.get(node_features_property, torch_zeros((0, 0))) for node_type in graph.ntypes
    }
    # Inference always runs in evaluation mode: training may end in training mode and a freshly built model starts
    # in it, and dropout must not make prediction scores stochastic. The published model is used only for inference.
    model.eval()
    with torch_no_grad():
        embeddings = model.online_forward(graph, graph_features)
    return embeddings


def score_pair(
    predictor,
    embeddings: dict[str, torch_Tensor],
    src_node: int,
    dest_node: int,
    src_type: str,
    dest_type: str,
) -> float:
    """Edge probability for one source/destination pair from precomputed embeddings.

    Args:
        predictor (torch.nn.Module): A reference to the predictor.
        embeddings (Dict[str, torch.Tensor]): Embeddings from compute_node_embeddings.
        src_node (int): Source node of the edge.
        dest_node (int): Destination node of the edge.
        src_type (str): Type of the source node.
        dest_type (str): Type of the destination node.

    Returns:
        float: Edge probability.
    """
    predictor.eval()
    with torch_no_grad():
        src_embedding = embeddings.get(src_type, torch_zeros((0, 0)))[src_node]
        dest_embedding = embeddings.get(dest_type, torch_zeros((0, 0)))[dest_node]
        score = predictor.forward_pred(src_embedding, dest_embedding)
        prob = torch_sigmoid(score)
    computed_return_value = prob.item()
    return computed_return_value
