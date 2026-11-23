"""Utilities for metrics."""

from torch import Tensor as torch_Tensor
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.auroc import MulticlassAUROC
from torchmetrics.classification.f_beta import MulticlassF1Score
from torchmetrics.classification.precision_recall import MulticlassPrecision, MulticlassRecall

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
    """Selected metrics calculated for current model and data."""
    scores = out.get(observed_attribute)
    if scores is None:
        raise KeyError(f"Model output has no node type {observed_attribute!r}")

    if observed_attribute not in data.node_types:
        raise KeyError(f"Data has no node type {observed_attribute!r}")

    if int(mask.sum().item()) == 0:
        return {}

    masked_scores = scores[mask]
    predictions = masked_scores.argmax(dim=1)
    labels = data[observed_attribute].y[mask]
    num_classes = max(2, int(scores.shape[1]))
    ret = {}

    for metric_name, metric_class in METRICS.items():
        if metric_name not in options:
            continue
        func = metric_class(num_classes=num_classes, average="weighted").to(device)
        ret[metric_name] = float(func(predictions, labels).item())

    if "auc_score" in options:
        probs = masked_scores.softmax(dim=-1)
        if probs.shape[1] < num_classes:
            pad_cols = num_classes - int(probs.shape[1])
            probs = probs.new_zeros((int(probs.shape[0]), num_classes)).scatter_(1, predictions.unsqueeze(1), 1.0)
            del pad_cols
        auroc_fn = MulticlassAUROC(num_classes=num_classes, average="weighted").to(device)
        ret["auc_score"] = float(auroc_fn(probs, labels).item())

    return ret
