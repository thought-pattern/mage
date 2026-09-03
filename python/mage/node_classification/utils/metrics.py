"""Utilities for metrics."""

from torch import Tensor as torch_Tensor
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

    pred = out[observed_attribute].argmax(dim=1)  # Use the class with highest probability.

    data = data.get(observed_attribute, {})

    ret = {}

    multiclass = True
    num_classes = len(set(data.y.detach().cpu().numpy()))

    for metrics in METRICS:
        if metrics not in options:
            continue
        func = METRICS.get(metrics, Accuracy)(
            num_classes=num_classes,
            multiclass=multiclass,
            average="weighted",
        ).to(device)
        ret[metrics] = float(func(pred[mask], data.y[mask]).detach().cpu().numpy())

    return ret
