"""
This class contains implementations of different layers
"""

from typing import Dict, List, Tuple

from numpy import array as np_array
from torch import Tensor as torch_Tensor
from torch import cat as torch_cat
from torch import concat as torch_concat
from torch import device as torch_device
from torch import nn
from torch import nn as torch_nn
from torch import rand as torch_rand
from torch import squeeze as torch_squeeze
from torch import sum as torch_sum
from torch import unsqueeze as torch_unsqueeze

from mage.tgn.definitions.tgn import GraphAttnDataType, GraphSumDataType
from mage.tgn.helper.simple_mlp import MLP


class TGNLayer(nn.Module):
    """
    Base class for all implementations
    """

    def __init__(
        self,
        embedding_dimension: int,
        edge_feature_dim: int,
        time_encoding_dim: int,
        node_features_dim: int,
        num_neighbors: int,
        num_of_layers: int,
        device: torch_device,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.edge_feature_dim = edge_feature_dim
        self.time_encoding_dim = time_encoding_dim
        self.node_features_dim = node_features_dim
        self.num_neighbors = num_neighbors
        self.num_of_layers = num_of_layers
        self.device = device


class TGNLayerGraphSumEmbedding(TGNLayer):
    """
    TGN layer implementation of graph sum embedding
    """

    def __init__(
        self,
        embedding_dimension: int,
        edge_feature_dim: int,
        time_encoding_dim: int,
        node_features_dim: int,
        num_neighbors: int,
        num_of_layers: int,
        device: torch_device,
    ):
        super().__init__(
            embedding_dimension,
            edge_feature_dim,
            time_encoding_dim,
            node_features_dim,
            num_neighbors,
            num_of_layers,
            device,
        )
        # Initialize W1 matrix and W2 matrix
        self.linear_1s = nn.ModuleList(
            torch_nn.Linear(
                embedding_dimension + time_encoding_dim + edge_feature_dim,
                embedding_dimension,
            )
            for _ in range(self.num_of_layers)
        ).to(self.device)

        self.linear_2s = nn.ModuleList(
            torch_nn.Linear(
                embedding_dimension + embedding_dimension, embedding_dimension
            )
            for _ in range(self.num_of_layers)
        ).to(self.device)
        self.relu = torch_nn.ReLU()

    def forward(self, data: GraphSumDataType):
        node_layers: List[List[Tuple[int, int]]]
        mappings: List[Dict[Tuple[int, int], int]]
        edge_layers: List[List[int]]
        neighbors_arr: List[List[Tuple[int, int]]]
        features: torch_Tensor
        edge_features: List[torch_Tensor]
        time_features: List[torch_Tensor]
        (
            node_layers,
            mappings,
            edge_layers,
            neighbors_arr,
            features,
            edge_features,
            time_features,
        ) = data

        out = features

        for k in range(self.num_of_layers):
            mapping = mappings[k]
            nodes = node_layers[k + 1]  # neighbors on next layer
            # represents how we globally gave index to node,timestamp mapping
            global_indexes = np_array([mappings[0].get((v, t), 0) for (v, t) in nodes])
            cur_neighbors = [
                neighbors_arr[index] for index in global_indexes
            ]  # neighbors and timestamps of nodes from next layer
            curr_edges = [edge_features[index] for index in global_indexes]
            curr_time = [time_features[index] for index in global_indexes]

            aggregate = self._aggregate(
                out, cur_neighbors, nodes, mapping, curr_edges, curr_time, k
            )

            curr_mapped_nodes = np_array(
                [mapping.get((v, t), False) for (v, t) in nodes]
            )

            concat_neigh_out = torch_cat((out[curr_mapped_nodes], aggregate), dim=1)
            out = self.linear_2s[k](concat_neigh_out)
        return out

    def _aggregate(
        self,
        features: torch_Tensor,
        rows: List[List[Tuple[int, int]]],
        nodes: List[Tuple[int, int]],
        mapping: Dict[Tuple[int, int], int],
        edge_features: List[torch_Tensor],
        time_features: List[torch_Tensor],
        layer: int,
    ) -> torch_Tensor:
        assert len(nodes) == len(rows)
        mapped_rows = [
            np_array([mapping.get((vi, ti), False) for (vi, ti) in row]) for row in rows
        ]
        out = torch_rand(len(nodes), self.embedding_dimension, device=self.device)

        # row represents list of indexes of "current neighbors" of edge_features on i-th index
        for i, row in enumerate(mapped_rows):
            features_curr = features[row][:]
            edge_feature_curr = edge_features[i][:]
            time_feature_curr = time_features[i][:]
            # shape(num_neighbors, embedding_dim+edge_features_dim+time_encoding_dim)
            # concatenation is done alongside columns, we have only 2 dimension, rows=0 and columns=1
            aggregate = torch_concat(
                (features_curr, edge_feature_curr, time_feature_curr), dim=1
            )

            # sum rows, but keep this dimension
            # shape = (1, embedding_dim+edge_features_dim+time_encoding_dim)
            aggregate_sum = torch_sum(aggregate, dim=0, keepdim=True)
            # shape = (1, embedding_dim)
            out_linear1 = self.linear_1s[layer](aggregate_sum)
            out_relu_linear1 = self.relu(out_linear1)

            out[i, :] = out_relu_linear1

        return out


class TGNLayerGraphAttentionEmbedding(TGNLayer):
    """
    TGN layer implementation of graph attention embedding
    """

    def __init__(
        self,
        embedding_dimension: int,
        edge_feature_dim: int,
        time_encoding_dim: int,
        node_features_dim: int,
        num_neighbors: int,
        num_of_layers: int,
        num_attention_heads: int,
        device: torch_device,
    ):
        super().__init__(
            embedding_dimension,
            edge_feature_dim,
            time_encoding_dim,
            node_features_dim,
            num_neighbors,
            num_of_layers,
            device,
        )

        self.query_dim = embedding_dimension + time_encoding_dim
        self.key_dim = embedding_dimension + edge_feature_dim + time_encoding_dim
        self.value_dim = self.key_dim
        self.num_attention_heads = num_attention_heads

        self.mlps = nn.ModuleList(
            MLP(
                [
                    self.query_dim + embedding_dimension,
                    embedding_dimension,
                    embedding_dimension,
                ]
            )
            for _ in range(self.num_of_layers)
        ).to(self.device)

        self.multi_head_attentions = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=self.query_dim,
                kdim=self.key_dim * self.num_neighbors,
                # set on neighbors num
                vdim=self.value_dim * self.num_neighbors,
                num_heads=num_attention_heads,  # this add as a parameter
                batch_first=True,
            )
            for _ in range(self.num_of_layers)
        ).to(self.device)  # this way no need to do torch.permute later <3

    def forward(self, data: GraphAttnDataType):
        node_layers: List[List[Tuple[int, int]]]
        mappings: List[Dict[Tuple[int, int], int]]
        edge_layers: List[List[int]]
        neighbors_arr: List[List[Tuple[int, int]]]
        features: torch_Tensor
        edge_features: List[torch_Tensor]
        time_features: List[torch_Tensor]
        time_encoder_zeros: torch_Tensor
        (
            node_layers,
            mappings,
            edge_layers,
            neighbors_arr,
            features,
            edge_features,
            time_features,
            time_encoder_zeros,
        ) = data

        out = features

        for k in range(self.num_of_layers):
            mapping = mappings[k]
            # shape = N
            nodes = node_layers[k + 1]  # neighbors on next layer
            # represents how we globally gave index to (node,timestamp) mapping
            global_indexes = np_array([mappings[0].get((v, t), 0) for (v, t) in nodes])
            cur_neighbors = [
                neighbors_arr[index] for index in global_indexes
            ]  # neighbors and timestamps of nodes from next layer
            curr_edges = [edge_features[index] for index in global_indexes]
            curr_time = [time_features[index] for index in global_indexes]

            # KEY_DIM = EMBEDDING_DIM + EDGE_FEATURES_DIM + TIME_ENC_DIM
            # shape = (N, NUM_NEIGHBORS * KEY_DIM)
            aggregate = self._aggregate(
                out, cur_neighbors, nodes, mapping, curr_edges, curr_time
            )

            # add third dimension,
            # shape = (1, N, NUM_NEIGHBORS * KEY_DIM)
            aggregate_unsqueeze = torch_unsqueeze(aggregate, dim=0)

            curr_mapped_nodes = np_array(
                [mapping.get((v, t), False) for (v, t) in nodes]
            )

            keys = aggregate_unsqueeze
            values = aggregate_unsqueeze
            # shape = (N, EMBEDDING_DIM + TIME_ENC_DIM)
            query_concat = torch_concat(
                (
                    out[curr_mapped_nodes],
                    time_encoder_zeros.repeat(len(curr_mapped_nodes), 1),
                ),
                dim=1,
            )
            # shape = (1, N, EMBEDDING_DIM + TIME_ENC_DIM)
            query = torch_unsqueeze(query_concat, dim=0)

            attn_out, _ = self.multi_head_attentions[k](
                query=query, key=keys, value=values
            )
            # shape = (N, EMBED_DIM + TIME_ENC_DIM)
            attn_out = torch_squeeze(attn_out)
            # shape = (N, EMBED_DIM + TIME_ENC_DIM + EMBED_DIM)
            concat_neigh_out = torch_cat((out[curr_mapped_nodes], attn_out), dim=1)
            # shape = (N, EMBED_DIM)
            out = self.mlps[k](concat_neigh_out)
        return out

    def _aggregate(
        self,
        features: torch_Tensor,
        rows: List[List[Tuple[int, int]]],
        nodes: List[Tuple[int, int]],
        mapping: Dict[Tuple[int, int], int],
        edge_features: List[torch_Tensor],
        time_features: List[torch_Tensor],
    ) -> torch_Tensor:
        assert len(nodes) == len(rows)
        mapped_rows = [
            np_array([mapping.get((vi, ti), False) for (vi, ti) in row]) for row in rows
        ]

        out = torch_rand(
            len(nodes), self.num_neighbors * self.key_dim, device=self.device
        )

        # row represents list of indexes of "current neighbors" of edge_features on i-th index
        for i, row in enumerate(mapped_rows):
            features_curr = features[row][:]
            edge_feature_curr = edge_features[i][:]
            time_feature_curr = time_features[i][:]

            # shape = (1, num_neighbors * (embedding_dim + edge_features_dim + time_encoding_dim)
            # after doing concatenation on columns side, reshape to have 1 row
            aggregate = torch_concat(
                (features_curr, edge_feature_curr, time_feature_curr), dim=1
            ).reshape((1, -1))  # -1 means to find dim by itself from matrix

            out[i, :] = aggregate

        return out
