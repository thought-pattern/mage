"""Utilities for DotPredictor."""

from typing import Dict

from dgl import function as fn
from dgl import graph as dgl_graph
from torch import Tensor as torch_Tensor
from torch import dot as torch_dot
from torch import nn

from mage.link_prediction.constants import Predictors


class DotPredictor(nn.Module):
    def forward(
        self,
        g: dgl_graph,
        node_embeddings: Dict[str, torch_Tensor],
        target_relation: str = "",
    ) -> torch_Tensor:
        """Prediction method of DotPredictor. It sets edge scores by calculating dot product
        between node neighbors.

        Args:
            g (dgl.graph): A reference to the graph.
            node_embeddings: (Dict[str, torch.Tensor]): Node embeddings for each node type.
            target_relation: str -> Unique edge type that is used for training.

        Returns:
            torch.Tensor: A tensor of edge scores.
        """
        with g.local_scope():
            for node_type in node_embeddings:  # Iterate over all node_types.
                g.nodes[node_type].data[Predictors.NODE_EMBEDDINGS] = (
                    node_embeddings.get(node_type, False)
                )

            # Compute a new edge feature named 'score' by a dot-product between the
            # embedding of source node and embedding of destination node.
            g.apply_edges(
                fn.u_dot_v(
                    Predictors.NODE_EMBEDDINGS,
                    Predictors.NODE_EMBEDDINGS,
                    Predictors.EDGE_SCORE,
                ),
                etype=target_relation,
            )
            scores = g.edata[Predictors.EDGE_SCORE]
            if not isinstance(scores, dict):
                _return_value = scores.view(-1)
                return _return_value
            if isinstance(
                target_relation, tuple
            ):  # Tuple[str, str, str] identification
                _return_value = scores[target_relation].view(-1)
                return _return_value
            if isinstance(target_relation, str):  # edge type identification
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
            float: Edge score.
        """
        _return_value = torch_dot(src_embedding, dest_embedding)
        return _return_value
