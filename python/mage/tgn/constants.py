"""Utilities for constants."""

from enum import Enum as enum_Enum


class TGNLayerType(enum_Enum):
    GraphSumEmbedding = "graph_sum"
    GraphAttentionEmbedding = "graph_attn"


class MessageFunctionType(enum_Enum):
    MLP = "mlp"
    Identity = "identity"


class MemoryUpdaterType(enum_Enum):
    GRU = "gru"
    RNN = "rnn"


class MessageAggregatorType(enum_Enum):
    Mean = "mean"
    Last = "last"
