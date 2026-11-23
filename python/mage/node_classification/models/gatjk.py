"""Utilities for gatjk."""

from mgp import List as mgp_List
from torch import Tensor as torch_Tensor, nn as torch_nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, Linear


class GATJK(torch_nn.Module):
    """Graph attention network with jumping knowledge for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_features_size: mgp_List[int],
        out_channels: int,
        heads: int = 3,
        dropout: float = 0.6,
        jk_type: str = "max",
    ):
        """Initialization of model."""
        super(GATJK, self).__init__()

        widths = list(hidden_features_size)
        if not widths or not all(isinstance(width, int) and not isinstance(width, bool) and width > 0 for width in widths):
            raise ValueError(f"GATJK needs at least one positive integer hidden width, received {hidden_features_size!r}")

        input_widths = [in_channels] + [width * heads for width in widths[:-1]]
        self.convs = torch_nn.ModuleList(
            GATConv(input_width, width, heads=heads, concat=True, add_self_loops=False)
            for input_width, width in zip(input_widths, widths, strict=True)
        )
        self.bns = torch_nn.ModuleList(torch_nn.BatchNorm1d(width * heads) for width in widths[:-1])

        target_width = widths[-1] * heads
        if jk_type in ("max", "lstm"):
            self.jk_projections = torch_nn.ModuleList(
                torch_nn.Identity() if width * heads == target_width else Linear(width * heads, target_width)
                for width in widths
            )
        else:
            self.jk_projections = torch_nn.ModuleList(torch_nn.Identity() for _ in widths)

        self.dropout = dropout
        self.dropout_layer = torch_nn.Dropout(p=dropout)
        self.activation = F.elu

        self.jump = JumpingKnowledge(jk_type, channels=target_width, num_layers=len(widths))
        if jk_type == "cat":
            self.final_project = Linear(sum(widths) * heads, out_channels)
        else:
            self.final_project = Linear(target_width, out_channels)

    def reset_parameters(self):
        """Reset of parameters."""
        for conv in self.convs:
            getattr(conv, "reset_parameters")()
        for bn in self.bns:
            getattr(bn, "reset_parameters")()
        for proj in self.jk_projections:
            reset_fn = getattr(proj, "reset_parameters", False)
            if callable(reset_fn):
                reset_fn()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()
        return False

    def forward(self, x: torch_Tensor, edge_index: torch_Tensor) -> torch_Tensor:
        """Forward passing of GATJK model."""
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(self.jk_projections[i](x))
            x = self.dropout_layer(x)
        x = self.convs[-1](x, edge_index)
        xs.append(self.jk_projections[-1](x))
        x = self.jump(xs)
        x = self.final_project(x)
        return x
