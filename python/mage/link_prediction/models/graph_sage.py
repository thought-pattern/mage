"""Utilities for graph sage."""

from typing import Dict, List

from dgl import graph as dgl_graph
from dgl.nn import HeteroGraphConv, SAGEConv
from torch import Tensor as torch_Tensor
from torch import device as torch_device
from torch import nn as torch_nn


class GraphSAGE(torch_nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_features_size: List[int],
        aggregator: str,
        feat_drops: List[float],
        edge_types: List[str],
        device: torch_device,
    ):
        """Initializes modules with sizes.

        Args:
            in_feats (int): Defines the size of the input features.
            hidden_features_size (List[int]): First element is the feature size and the rest specifies layer size.
            aggregator (str):  Aggregator used in models. Can be one of the following: lstm, gcn, mean and pool.
            feat_drops (List[float]): Features dropout rate for each layer.
            edge_types (List[str]): All edge types that are occurring in the heterogeneous network.
        """
        super(GraphSAGE, self).__init__()
        self.layers = torch_nn.ModuleList()
        self.num_layers = len(hidden_features_size)
        # Define activations
        activations = [
            torch_nn.functional.relu for _ in range(self.num_layers - 1)
        ]  # All activations except last layer
        activations.append(False)
        # Create layers
        for i in range(self.num_layers):
            sage_layer = SAGEConv(
                in_feats=in_feats,
                out_feats=hidden_features_size[i],
                aggregator_type=aggregator,
                feat_drop=feat_drops[i],
                activation=activations[i],
            ).to(device)
            self.layers.append(
                HeteroGraphConv(
                    dict.fromkeys(edge_types, sage_layer), aggregate="sum"
                ).to(device)
            )
            in_feats = hidden_features_size[i]

    def forward(
        self, blocks: List[dgl_graph], h: Dict[str, torch_Tensor]
    ) -> Dict[str, torch_Tensor]:
        (
            "Performs forward pass on batches.\n\n        Args:\n            blocks (List[dgl.heterograph.DG"  # Continue literal.
            "LBlock]): First block is DGLBlock of all nodes that are needed to compute representations fo"  # Continue literal.
            "r second block. Second block is sampled graph.\n            h (Dict[str, torch.Tensor]): Inpu"  # Continue literal.
            "t features for every node type.\n\n        Returns:\n            Dict[str, torch.Tensor]: Embed"  # Continue literal.
            "dings for every node type.\n"
        )
        for index, layer in enumerate(self.layers):
            h = layer(blocks[index], h)

        return h

    def online_forward(
        self, graph: dgl_graph, h: Dict[str, torch_Tensor]
    ) -> Dict[str, torch_Tensor]:
        """Performs forward pass on batches.

        Args:
            graph (dgl.heterograph): Whole graph instance used in prediction.
            h (Dict[str, torch.Tensor]): Input features for every node type.

        Returns:
            Dict[str, torch.Tensor]: Embeddings for every node type.
        """
        for layer in self.layers:
            h = layer(graph, h)

        return h
