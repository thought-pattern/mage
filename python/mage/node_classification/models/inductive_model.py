"""Utilities for inductive model."""

from mgp import List as mgp_List
from torch import nn as torch_nn
from torch import Tensor as torch_Tensor
from torch.nn import functional as F
from torch_geometric import nn as torch_geometric_nn


class InductiveModel(torch_nn.Module):
    def __init__(
        self,
        layer_type: str,
        in_channels: int,
        hidden_features_size: mgp_List[int],
        out_channels: int,
        aggr: str,
    ):
        """Initialization of model.

        Args:
            layer_type (str): type of layer
            in_channels (int): dimension of input channels
            hidden_features_size (mgp.List[int]): list of dimensions of hidden features
            out_channels (int): dimension of output channels
            aggr (str): aggregator type
        """

        super(InductiveModel, self).__init__()

        self.convs = torch_nn.ModuleList()
        self.bns = torch_nn.ModuleList()

        conv = getattr(torch_geometric_nn, layer_type + "Conv")
        if len(hidden_features_size) > 0:
            self.convs.append(conv(in_channels, hidden_features_size[0], aggr=aggr))
            self.bns.append(torch_nn.BatchNorm1d(hidden_features_size[0]))
            for i in range(len(hidden_features_size) - 1):
                self.convs.append(
                    conv(
                        hidden_features_size[i], hidden_features_size[i + 1], aggr=aggr
                    )
                )
                self.bns.append(torch_nn.BatchNorm1d(hidden_features_size[i + 1]))
            self.convs.append(conv(hidden_features_size[-1], out_channels, aggr=aggr))
        else:
            self.convs.append(conv(in_channels, out_channels, aggr=aggr))

    def forward(self, x: torch_Tensor, edge_index: torch_Tensor) -> torch_Tensor:
        """Forward propagation

        Args:
            x (torch.tensor): matrix of embeddings
            edge_index (torch.tensor): matrix of edges

        Returns:
            torch.tensor: embeddings after last layer of network is applied
        """

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)

            # apply relu and dropout on all layers except last one
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)

        return x
