"""Utilities for memory updater."""

from typing import Tuple

from torch import Tensor as torch_Tensor
from torch import device as torch_device
from torch import nn


class MemoryUpdater(nn.Module):
    """
    This is base class for memory updater implementation
    """

    def __init__(
        self, memory_dimension: int, message_dimension: int, device: torch_device
    ):
        super().__init__()
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device


class MemoryUpdaterGRU(MemoryUpdater):
    def __init__(
        self, memory_dimension: int, message_dimension: int, device: torch_device
    ):
        super().__init__(memory_dimension, message_dimension, device)

        self.memory_updater_net = nn.GRUCell(
            input_size=message_dimension, hidden_size=memory_dimension
        ).to(self.device)

    def forward(self, data: Tuple[torch_Tensor, torch_Tensor]):
        # messages shape = (1, message_dim)
        # memory shape = (memory_dim,)
        messages, memory = data

        # memory_dim = (1, memory_dim)
        memory = memory.unsqueeze(0)

        _return_value = self.memory_updater_net(messages, memory)
        return _return_value


class MemoryUpdaterRNN(MemoryUpdater):
    def __init__(
        self, memory_dimension: int, message_dimension: int, device: torch_device
    ):
        super().__init__(memory_dimension, message_dimension, device)

        self.memory_updater_net = nn.RNNCell(
            input_size=message_dimension, hidden_size=memory_dimension
        ).to(self.device)

    def forward(self, data: Tuple[torch_Tensor, torch_Tensor]):
        # messages shape = (1, message_dim)
        # memory shape = (memory_dim,)
        messages, memory = data

        # memory_dim = (1, memory_dim)
        memory = memory.unsqueeze(0)

        _return_value = self.memory_updater_net(messages, memory)
        return _return_value
