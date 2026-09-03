"""Utilities for graph sage."""

from importlib import import_module

from torch import Tensor as torch_Tensor
from torch import device as torch_device
from torch import nn as torch_nn


class GraphSAGE(torch_nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_features_size: list[int],
        aggregator: str,
        feat_drops: list[float],
        edge_types: list[str],
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
        if not hidden_features_size:
            raise ValueError("GraphSAGE requires at least one hidden layer")
        if len(feat_drops) != len(hidden_features_size):
            raise ValueError(
                f"Expected one feature-drop rate per layer, received {len(feat_drops)} for {len(hidden_features_size)} layers"
            )
        if not edge_types:
            raise ValueError("GraphSAGE requires at least one edge type")
        if aggregator not in {"lstm", "gcn", "mean", "pool"}:
            raise ValueError(f"Unsupported GraphSAGE aggregator: {aggregator}")
        if any(drop < 0.0 or drop >= 1.0 for drop in feat_drops):
            raise ValueError(f"Feature-drop rates must be in [0, 1), received {feat_drops}")

        try:
            dgl_nn = import_module("dgl.nn")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("GraphSAGE requires the DGL package") from error
        hetero_graph_conv = getattr(dgl_nn, "HeteroGraphConv", False)
        sage_conv = getattr(dgl_nn, "SAGEConv", False)
        if not callable(hetero_graph_conv) or not callable(sage_conv):
            raise ImportError("DGL does not provide HeteroGraphConv and SAGEConv")

        super(GraphSAGE, self).__init__()
        self.layers = torch_nn.ModuleList()
        self.num_layers = len(hidden_features_size)
        # Define activations
        # Create layers
        for i in range(self.num_layers):
            layer_parameters = {
                "in_feats": in_feats,
                "out_feats": hidden_features_size[i],
                "aggregator_type": aggregator,
                "feat_drop": feat_drops[i],
            }
            if i < self.num_layers - 1:
                layer_parameters["activation"] = torch_nn.functional.relu
            sage_layer = sage_conv(**layer_parameters)
            if not isinstance(sage_layer, torch_nn.Module):
                raise TypeError(f"DGL SAGEConv returned {type(sage_layer)}, expected torch.nn.Module")
            sage_layer = sage_layer.to(device)
            heterogeneous_layer = hetero_graph_conv(dict.fromkeys(edge_types, sage_layer), aggregate="sum")
            if not isinstance(heterogeneous_layer, torch_nn.Module):
                raise TypeError(f"DGL HeteroGraphConv returned {type(heterogeneous_layer)}, expected torch.nn.Module")
            self.layers.append(heterogeneous_layer.to(device))
            in_feats = hidden_features_size[i]

    def forward(
        self, blocks: list[object], h: dict[str, torch_Tensor]
    ) -> dict[str, torch_Tensor]:
        (
            "Performs forward pass on batches.\n\n        Args:\n            blocks (List[dgl.heterograph.DG"  # Continue literal.
            "LBlock]): First block is DGLBlock of all nodes that are needed to compute representations fo"  # Continue literal.
            "r second block. Second block is sampled graph.\n            h (Dict[str, torch.Tensor]): Inpu"  # Continue literal.
            "t features for every node type.\n\n        Returns:\n            Dict[str, torch.Tensor]: Embed"  # Continue literal.
            "dings for every node type.\n"
        )
        if len(blocks) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} graph blocks, received {len(blocks)}")
        for index, layer in enumerate(self.layers):
            layer_output = layer(blocks[index], h)
            if not isinstance(layer_output, dict):
                raise TypeError(f"DGL GraphSAGE layer returned {type(layer_output)}, expected dict")
            h = layer_output

        return h

    def online_forward(
        self, graph: object, h: dict[str, torch_Tensor]
    ) -> dict[str, torch_Tensor]:
        """Performs forward pass on batches.

        Args:
            graph (dgl.heterograph): Whole graph instance used in prediction.
            h (Dict[str, torch.Tensor]): Input features for every node type.

        Returns:
            Dict[str, torch.Tensor]: Embeddings for every node type.
        """
        for layer in self.layers:
            layer_output = layer(graph, h)
            if not isinstance(layer_output, dict):
                raise TypeError(f"DGL GraphSAGE layer returned {type(layer_output)}, expected dict")
            h = layer_output

        return h
