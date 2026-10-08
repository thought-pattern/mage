"""Utilities for MLPPredictor."""

from dgl import graph as dgl_graph
from torch import Tensor as torch_Tensor, cat as torch_cat, device as torch_device, nn as torch_nn
from torch.nn import functional as F

from mage.link_prediction.constants import Predictors


class MLPPredictor(torch_nn.Module):
    def __init__(self, h_feats: int, device: torch_device) -> None:
        super().__init__()

        self.W1 = torch_nn.Linear(h_feats * 2, h_feats, device=device)
        self.W2 = torch_nn.Linear(h_feats, 1, device=device)

    def apply_edges(self, edges) -> dict:
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
        computed_return_value = {Predictors.EDGE_SCORE: self.W2(F.relu(self.W1(h))).squeeze(1)}
        return computed_return_value

    def forward(
        self,
        g: dgl_graph,
        node_embeddings: dict[str, torch_Tensor],
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
            for node_type, embedding in node_embeddings.items():  # Iterate over all node_types.
                g.nodes[node_type].data[Predictors.NODE_EMBEDDINGS] = embedding

            g.apply_edges(self.apply_edges, etype=target_relation)
            scores = g.edata[Predictors.EDGE_SCORE]

            if not isinstance(scores, dict):
                computed_return_value = scores.view(-1)
                return computed_return_value

            # With several edge types DGL returns the scores keyed by canonical relation. A canonical (source, edge,
            # destination) target is admitted by exact key, an edge type name by its middle element; a target with no
            # scores falls through to the error below.
            for key, val in scores.items():
                if (isinstance(target_relation, tuple) and key == target_relation) or (
                    isinstance(target_relation, str) and key[1] == target_relation
                ):
                    computed_return_value = val.view(-1)
                    return computed_return_value
        raise ValueError("DGL did not return scores for the requested edge relation")

    def forward_pred(self, src_embedding: torch_Tensor, dest_embedding: torch_Tensor) -> torch_Tensor:
        """Efficient implementation for predict method of MLPPredictor.

        Args:
            src_embedding (torch.Tensor): Embedding of the source node.
            dest_embedding (torch.Tensor): Embedding of the destination node.

        Returns:
            torch.Tensor: Edge score computed (a one-element tensor).
        """
        h = torch_cat([src_embedding, dest_embedding])
        computed_return_value = self.W2(F.relu(self.W1(h)))
        return computed_return_value
