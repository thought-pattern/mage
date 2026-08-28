"""Utilities for MLPPredictor."""

from typing import Dict, Tuple

from dgl import graph as dgl_graph
from torch import Tensor as torch_Tensor
from torch import cat as torch_cat
from torch import device as torch_device
from torch import nn as torch_nn
from torch.nn import functional as F

from mage.link_prediction.constants import Predictors


class MLPPredictor(torch_nn.Module):
    def __init__(self, h_feats: int, device: torch_device) -> None:
        super().__init__()

        self.W1 = torch_nn.Linear(h_feats * 2, h_feats, device=device)
        self.W2 = torch_nn.Linear(h_feats, 1, device=device)

    def apply_edges(self, edges: Tuple[torch_Tensor, torch_Tensor]) -> Dict:
        (
            "Computes a scalar score for each edge of the given graph.\n\n        Args:\n            edges ("  # Continue literal.
            "Tuple[torch.Tensor, torch.Tensor]): Has three members: ``src``, ``dst`` and ``data``, each o"  # Continue literal.
            "f which is a dictionary representing the features of the source nodes, the destination nodes"  # Continue literal.
            " and the edges themselves.\n        Returns:\n            Dict: A dictionary of new edge featu"  # Continue literal.
            "res\n"
        )
        h = torch_cat(
            [
                edges.src[Predictors.NODE_EMBEDDINGS],
                edges.dst[Predictors.NODE_EMBEDDINGS],
            ],
            1,
        )
        _return_value = {Predictors.EDGE_SCORE: self.W2(F.relu(self.W1(h))).squeeze(1)}
        return _return_value

    def forward(
        self,
        g: dgl_graph,
        node_embeddings: Dict[str, torch_Tensor],
        target_relation: str = "",
    ) -> torch_Tensor:
        """Calculates forward pass of MLPPredictor.

        Args:
            g (dgl.graph): A reference to the graph for which edge scores will be computed.
            node_embeddings (Dict[str, torch.Tensor]): node embeddings for each node type.
            target_relation: str -> Unique edge type that is used for training.
        Returns:
            torch.Tensor: A tensor of edge scores.
        """
        with g.local_scope():
            for node_type in node_embeddings:  # Iterate over all node_types.
                g.nodes[node_type].data[Predictors.NODE_EMBEDDINGS] = (
                    node_embeddings.get(node_type, False)
                )

            g.apply_edges(self.apply_edges, etype=target_relation)
            scores = g.edata[Predictors.EDGE_SCORE]

            if not isinstance(scores, dict):
                _return_value = scores.view(-1)
                return _return_value

            if isinstance(
                target_relation, tuple
            ):  # Tuple[str, str, str] identification
                _return_value = scores[target_relation].view(-1)
                return _return_value

            if isinstance(target_relation, str):
                for key, val in scores.items():
                    if key[1] == target_relation:
                        _return_value = val.view(-1)
                        return _return_value
        return False

    def forward_pred(
        self, src_embedding: torch_Tensor, dest_embedding: torch_Tensor
    ) -> float:
        """Efficient implementation for predict method of DotPredictor.

        Args:
            src_embedding (torch.Tensor): Embedding of the source node.
            dest_embedding (torch.Tensor): Embedding of the destination node.

        Returns:
            float: Edge score computed.
        """
        h = torch_cat([src_embedding, dest_embedding])
        _return_value = self.W2(F.relu(self.W1(h)))
        return _return_value
