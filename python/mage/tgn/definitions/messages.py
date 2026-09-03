"""Utilities for messages."""

from torch import Tensor as torch_Tensor


class RawMessage:
    """
    Raw Message class is a container for information needed to compute message from Node-wise event and Interaction-wise
    event respectfully.
    """

    def __init__(self, source: int, timestamp: int):
        super().__init__()
        self.source = source
        self.timestamp = timestamp

    def detach_memory(self) -> bool:
        raise NotImplementedError("RawMessage subclasses must implement detach_memory")

    def __str__(self):
        computed_return_value = "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)
        return computed_return_value


class NodeRawMessage(RawMessage):
    def __init__(
        self,
        source_memory: torch_Tensor,
        timestamp: int,
        node_features: torch_Tensor,
        source: int,
    ):
        super().__init__(source, timestamp)
        self.source_memory = source_memory
        self.node_features = node_features

    def detach_memory(self) -> bool:
        if self.source_memory.grad is not None:
            self.source_memory.detach_()
            self.source_memory.zero_()
        if self.node_features.grad is not None:
            self.node_features.detach_()
            self.node_features.zero_()
        return False


class InteractionRawMessage(RawMessage):
    """
    Interaction Raw Message is created on Interaction Event - just a fancy name for creation of new
    edge in a graph.
    It consists of memory of the source node, memory of the destination node, difference
    in time (delta time) of current time and last interaction for source node, and edge features

    """

    def __init__(
        self,
        source_memory: torch_Tensor,
        dest_memory: torch_Tensor,
        delta_time: torch_Tensor,
        edge_features: torch_Tensor,
        source: int,
        timestamp: int,
    ):
        super().__init__(source, timestamp)
        self.source_memory = source_memory
        self.dest_memory = dest_memory
        self.delta_time = delta_time
        self.edge_features = edge_features

    def detach_memory(self) -> bool:
        if self.source_memory.grad is not None:
            self.source_memory.detach_()
            self.source_memory.zero_()
        if self.dest_memory.grad is not None:
            self.dest_memory.detach_()
            self.dest_memory.zero_()
        if self.delta_time.grad is not None:
            self.delta_time.detach_()
            self.delta_time.zero_()
        if self.edge_features.grad is not None:
            self.edge_features.detach_()
            self.edge_features.zero_()
        return False
