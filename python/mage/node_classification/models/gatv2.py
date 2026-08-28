"""Utilities for gatv2."""

from mgp import List as mgp_List

from mage.node_classification.models.inductive_model import InductiveModel


class GATv2(InductiveModel):
    def __init__(
        self,
        in_channels: int,
        hidden_features_size: mgp_List[int],
        out_channels: int,
        aggr: str,
    ):
        super().__init__(
            layer_type="GATv2",
            in_channels=in_channels,
            hidden_features_size=hidden_features_size,
            out_channels=out_channels,
            aggr=aggr,
        )
