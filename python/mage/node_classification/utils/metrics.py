"""Utilities for metrics."""

from torch import Tensor as torch_Tensor, zeros as torch_zeros
from torchmetrics import AUC, Accuracy, F1Score, Precision, Recall

METRICS = {
    "accuracy": Accuracy,
    "auc_score": AUC,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
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
    if observed_attribute not in out:
        raise KeyError(f"Model output has no node type {observed_attribute!r}")
    pred = out.get(observed_attribute, torch_zeros((0, 0))).argmax(dim=1)  # Use the class with highest probability.

    # node stores are selected by subscription; HeteroData.get reads the global store
    if observed_attribute not in data.node_types:
        raise KeyError(f"Data has no node type {observed_attribute!r}")
    data = data[observed_attribute]

    ret = {}

    multiclass = True
    num_classes = len(set(data.y.detach().cpu().numpy()))

    for metric_name, metric_class in METRICS.items():
        if metric_name not in options:
            continue
        func = metric_class(
            num_classes=num_classes,
            multiclass=multiclass,
            average="weighted",
        ).to(device)
        ret[metric_name] = float(func(pred[mask], data.y[mask]).detach().cpu().numpy())

    return ret
