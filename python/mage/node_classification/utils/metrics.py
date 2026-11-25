"""Utilities for metrics."""

from torch import Tensor as torch_Tensor
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from torchmetrics.classification.precision_recall import MulticlassPrecision, MulticlassRecall
from torchmetrics.utilities.compute import auc

METRICS = {
    "accuracy": MulticlassAccuracy,
    "precision": MulticlassPrecision,
    "recall": MulticlassRecall,
    "f1_score": MulticlassF1Score,
}


def metrics(
    mask: torch_Tensor,
    out,
    data,
    options: list[str],
    observed_attribute: str,
    device: str,
) -> dict[str, float]:
    """Selected metrics calculated for current model and data.

    Args:
        mask (torch.tensor): used to mask which embeddings should be used
        out (torch.tensor): output of the model
        data (Data): dataset variable
        options (List[str]): list of options to be calculated
        device (str): cpu or cuda

    Returns:
        Dict: dictionary of calculated metrics
    """

    # The heterogeneous model returns one output tensor per node type; the observed type must be among them.
    scores = out.get(observed_attribute)
    if scores is None:
        raise KeyError(f"Model output has no node type {observed_attribute!r}")

    # node stores are selected by subscription; HeteroData.get reads the global store
    if observed_attribute not in data.node_types:
        raise KeyError(f"Data has no node type {observed_attribute!r}")

    predictions = scores.argmax(dim=1)[mask]
    labels = data[observed_attribute].y[mask]
    # TorchMetrics requires at least two classes. An absent class has zero weight.
    num_classes = max(2, scores.shape[1])
    ret = {}

    for metric_name, metric_class in METRICS.items():
        if metric_name not in options:
            continue
        func = metric_class(num_classes=num_classes, average="weighted").to(device)
        ret[metric_name] = float(func(predictions, labels).item())

    if "auc_score" in options:
        ret["auc_score"] = float(auc(predictions, labels).item())

    return ret
