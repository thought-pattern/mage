"""Utilities for gatjk."""

from mgp import List as mgp_List
from torch import nn as torch_nn
from torch import Tensor as torch_Tensor
from torch.nn import functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, Linear


class GATJK(torch_nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_features_size: mgp_List[int],
        out_channels: int,
        heads: int = 3,
        dropout: float = 0.6,
        jk_type: str = "max",
    ):
        """Initialization of model.

        Args:
            in_channels (int): dimension of input channels
            hidden_features_size (mgp.List[int]): list of dimensions of every hidden channel
            out_channels (int): dimension of output channels
            heads (int): number of heads for multi-head attention used for regularisation (https://petar-v.com/GAT/)
            dropout (float): ratio of layer outputs which are randomly ignored during training
            jk_type (str): type of aggregation mechanism for Jumping Knowledge Network as in
                            Representation Learning on Graphs with Jumping Knowledge Networks by K. Xu et al.
        """

        super(GATJK, self).__init__()

        # Layer i maps the previous width (the input, then each earlier hidden width times the concatenated heads) to
        # hidden_features_size[i], and every layer except the last is batch-normalised, so a single hidden width is a
        # complete one-layer architecture.
        widths = list(hidden_features_size)
        if not widths or not all(isinstance(width, int) and not isinstance(width, bool) and width > 0 for width in widths):
            raise ValueError(f"GATJK needs at least one positive integer hidden width, received {hidden_features_size!r}")
        # max and lstm aggregation combine the layer outputs elementwise, so every layer must have the same width.
        if jk_type in ("max", "lstm") and len(set(widths)) != 1:
            raise ValueError(f"GATJK with jk_type {jk_type!r} needs equal hidden widths, received {widths!r}")

        input_widths = [in_channels] + [width * heads for width in widths[:-1]]
        self.convs = torch_nn.ModuleList(
            GATConv(input_width, width, heads=heads, concat=True, add_self_loops=False)
            for input_width, width in zip(input_widths, widths, strict=True)
        )
        self.bns = torch_nn.ModuleList(torch_nn.BatchNorm1d(width * heads) for width in widths[:-1])

        self.dropout = dropout
        self.activation = F.elu  # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=widths[-1] * heads, num_layers=len(widths))
        if jk_type == "cat":
            self.final_project = Linear(sum(widths) * heads, out_channels)
        else:  # max or lstm
            self.final_project = Linear(widths[-1] * heads, out_channels)

    def reset_parameters(self):
        """Reset of parameters."""
        for conv in self.convs:
            getattr(conv, "reset_parameters")()
        for bn in self.bns:
            getattr(bn, "reset_parameters")()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()
        return False

    def forward(self, x: torch_Tensor, edge_index: torch_Tensor) -> torch_Tensor:
        """Forward passing of GATJK model.

        Args:
            x (torch.tensor): input features
            edge_index (torch.tensor): edge indices

        Returns:
            torch.tensor: embeddings after last layer of network is applied
        """
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x
