"""Utilities for inductive model."""

from mgp import List as mgp_List
from torch import Tensor as torch_Tensor, nn as torch_nn
from torch_geometric import nn as torch_geometric_nn


class InductiveModel(torch_nn.Module):
    """Inductive node classification model built from the configured layer type."""

    def __init__(
        self,
        layer_type: str,
        in_channels: int,
        hidden_features_size: mgp_List[int],
        out_channels: int,
        aggr: str,
    ):
        """Initialization of model."""
        super(InductiveModel, self).__init__()

        self.convs = torch_nn.ModuleList()
        self.bns = torch_nn.ModuleList()
        self.dropout_layer = torch_nn.Dropout(p=0.5)

        conv = getattr(torch_geometric_nn, layer_type + "Conv")
        if len(hidden_features_size) > 0:
            self.convs.append(conv(in_channels, hidden_features_size[0], aggr=aggr))
            self.bns.append(torch_nn.BatchNorm1d(hidden_features_size[0]))
            for i in range(len(hidden_features_size) - 1):
                self.convs.append(conv(hidden_features_size[i], hidden_features_size[i + 1], aggr=aggr))
                self.bns.append(torch_nn.BatchNorm1d(hidden_features_size[i + 1]))
            self.convs.append(conv(hidden_features_size[-1], out_channels, aggr=aggr))
        else:
            self.convs.append(conv(in_channels, out_channels, aggr=aggr))

    def forward(self, x: torch_Tensor, edge_index: torch_Tensor) -> torch_Tensor:
        """Forward propagation."""
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = x.relu()
                x = self.dropout_layer(x)

        return x
