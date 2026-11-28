"""
This class contains implementations of different layers
"""

from numpy import array as np_array
from torch import (
    Tensor as torch_Tensor,
    bool as torch_bool,
    cat as torch_cat,
    concat as torch_concat,
    device as torch_device,
    nn,
    ones as torch_ones,
    sum as torch_sum,
    zeros as torch_zeros,
)

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
            nn.Linear(
                embedding_dimension + time_encoding_dim + edge_feature_dim,
                embedding_dimension,
            )
            for _ in range(self.num_of_layers)
        ).to(self.device)

        self.linear_2s = nn.ModuleList(
            nn.Linear(embedding_dimension + embedding_dimension, embedding_dimension) for _ in range(self.num_of_layers)
        ).to(self.device)
        self.relu = nn.ReLU()

    def forward(self, data: tuple) -> torch_Tensor:
        if len(data) != 7:
            raise ValueError(f"Graph-sum data must contain seven values, received {len(data)}")
        node_layers: list[list[tuple[int, int]]]
        mappings: list[dict[tuple[int, int], int]]
        neighbors_arr: list[list[tuple[int, int]]]
        features: torch_Tensor
        edge_features: list[torch_Tensor]
        time_features: list[torch_Tensor]
        (
            node_layers,
            mappings,
            _,
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
            global_indexes = []
            for node in nodes:
                index = mappings[0].get(node, -1)
                if index < 0:
                    raise KeyError(f"Node {node} is missing from the global TGN mapping")
                global_indexes.append(index)
            cur_neighbors = [neighbors_arr[index] for index in global_indexes]
            curr_edges = [edge_features[index] for index in global_indexes]
            curr_time = [time_features[index] for index in global_indexes]

            aggregate = self.internal_aggregate(out, cur_neighbors, nodes, mapping, curr_edges, curr_time, k)

            mapped_nodes = []
            for node in nodes:
                mapped_node = mapping.get(node, -1)
                if mapped_node < 0:
                    raise KeyError(f"Node {node} is missing from the current TGN mapping")
                mapped_nodes.append(mapped_node)
            curr_mapped_nodes = np_array(mapped_nodes)

            concat_neigh_out = torch_cat((out[curr_mapped_nodes], aggregate), dim=1)
            out = self.linear_2s[k](concat_neigh_out)
        return out

    def internal_aggregate(
        self,
        features: torch_Tensor,
        rows: list[list[tuple[int, int]]],
        nodes: list[tuple[int, int]],
        mapping: dict[tuple[int, int], int],
        edge_features: list[torch_Tensor],
        time_features: list[torch_Tensor],
        layer: int,
    ) -> torch_Tensor:
        if len(nodes) != len(rows):
            raise ValueError(f"Node and neighborhood counts differ: {len(nodes)} != {len(rows)}")
        mapped_rows = []
        for row in rows:
            mapped_row = []
            for node in row:
                mapped_node = mapping.get(node, -1)
                if mapped_node < 0:
                    raise KeyError(f"Neighbor {node} is missing from the TGN mapping")
                mapped_row.append(mapped_node)
            # a row lists only sampled neighbors and may be empty; integer dtype keeps an empty row a valid index
            mapped_rows.append(np_array(mapped_row, dtype=int))
        out = torch_zeros(len(nodes), self.embedding_dimension, device=self.device)

        # row represents list of indexes of "current neighbors" of edge_features on i-th index
        for i, row in enumerate(mapped_rows):
            features_curr = features[row][:]
            edge_feature_curr = edge_features[i][:]
            time_feature_curr = time_features[i][:]
            # shape(num_neighbors, embedding_dim+edge_features_dim+time_encoding_dim)
            # concatenation is done alongside columns, we have only 2 dimension, rows=0 and columns=1
            aggregate = torch_concat((features_curr, edge_feature_curr, time_feature_curr), dim=1)

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

        if num_attention_heads < 1 or self.query_dim % num_attention_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be positive and divide query_dim ({self.query_dim})"
            )

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
                kdim=self.key_dim,
                vdim=self.value_dim,
                num_heads=num_attention_heads,
                batch_first=True,
            )
            for _ in range(self.num_of_layers)
        ).to(self.device)

    def forward(self, data: tuple) -> torch_Tensor:
        if len(data) != 8:
            raise ValueError(f"Graph-attention data must contain eight values, received {len(data)}")
        node_layers: list[list[tuple[int, int]]]
        mappings: list[dict[tuple[int, int], int]]
        neighbors_arr: list[list[tuple[int, int]]]
        features: torch_Tensor
        edge_features: list[torch_Tensor]
        time_features: list[torch_Tensor]
        time_encoder_zeros: torch_Tensor
        (
            node_layers,
            mappings,
            _,
            neighbors_arr,
            features,
            edge_features,
            time_features,
            time_encoder_zeros,
        ) = data

        if tuple(time_encoder_zeros.shape) != (1, self.time_encoding_dim):
            raise ValueError(
                f"Zero-time encoding must have shape (1, {self.time_encoding_dim}); received {tuple(time_encoder_zeros.shape)}"
            )

        out = features

        for k in range(self.num_of_layers):
            mapping = mappings[k]
            nodes = node_layers[k + 1]
            global_indexes = []
            for node in nodes:
                index = mappings[0].get(node, -1)
                if index < 0:
                    raise KeyError(f"Node {node} is missing from the global TGN mapping")
                global_indexes.append(index)
            cur_neighbors = [neighbors_arr[index] for index in global_indexes]
            curr_edges = [edge_features[index] for index in global_indexes]
            curr_time = [time_features[index] for index in global_indexes]

            # Each target node i is its own batch element (N, num_neighbors, key_dim) so neighborhoods never mix across nodes.
            keys, padding_mask, has_neighbors = self.internal_aggregate(out, cur_neighbors, nodes, mapping, curr_edges, curr_time)
            values = keys

            mapped_nodes = []
            for node in nodes:
                mapped_node = mapping.get(node, -1)
                if mapped_node < 0:
                    raise KeyError(f"Node {node} is missing from the current TGN mapping")
                mapped_nodes.append(mapped_node)
            curr_mapped_nodes = np_array(mapped_nodes)

            query_concat = torch_concat(
                (
                    out[curr_mapped_nodes],
                    time_encoder_zeros.repeat(len(curr_mapped_nodes), 1),
                ),
                dim=1,
            )
            query = query_concat.unsqueeze(1)

            attn_out = torch_zeros(len(curr_mapped_nodes), self.query_dim, device=self.device)
            if bool(has_neighbors.any()):
                active_attn, _ = self.multi_head_attentions[k](
                    query=query[has_neighbors],
                    key=keys[has_neighbors],
                    value=values[has_neighbors],
                    key_padding_mask=padding_mask[has_neighbors],
                )
                attn_out[has_neighbors] = active_attn.squeeze(1)

            concat_neigh_out = torch_cat((out[curr_mapped_nodes], attn_out), dim=1)
            out = self.mlps[k](concat_neigh_out)
        return out

    def internal_aggregate(
        self,
        features: torch_Tensor,
        rows: list[list[tuple[int, int]]],
        nodes: list[tuple[int, int]],
        mapping: dict[tuple[int, int], int],
        edge_features: list[torch_Tensor],
        time_features: list[torch_Tensor],
    ) -> tuple[torch_Tensor, torch_Tensor, torch_Tensor]:
        if len(nodes) != len(rows):
            raise ValueError(f"Node and neighborhood counts differ: {len(nodes)} != {len(rows)}")
        mapped_rows = []
        for row in rows:
            mapped_row = []
            for node in row:
                mapped_node = mapping.get(node, -1)
                if mapped_node < 0:
                    raise KeyError(f"Neighbor {node} is missing from the TGN mapping")
                mapped_row.append(mapped_node)
            mapped_rows.append(np_array(mapped_row, dtype=int))

        out = torch_zeros(len(nodes), self.num_neighbors, self.key_dim, device=self.device)
        padding_mask = torch_ones(len(nodes), self.num_neighbors, dtype=torch_bool, device=self.device)
        has_neighbors = torch_zeros(len(nodes), dtype=torch_bool, device=self.device)

        for i, row in enumerate(mapped_rows):
            neighbor_count = len(row)
            if neighbor_count == 0:
                continue
            features_curr = features[row][:]
            edge_feature_curr = edge_features[i][:]
            time_feature_curr = time_features[i][:]
            aggregate = torch_concat((features_curr, edge_feature_curr, time_feature_curr), dim=1)
            out[i, :neighbor_count, :] = aggregate
            padding_mask[i, :neighbor_count] = False
            has_neighbors[i] = True

        return out, padding_mask, has_neighbors
