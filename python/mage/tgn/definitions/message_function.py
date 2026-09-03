"""Utilities for message function."""

from torch import concat as torch_concat
from torch import device as torch_device
from torch import nn


class MessageFunction(nn.Module):
    """
    This is base class for Message function implementation
    """

    def __init__(self, raw_message_dimension: int, message_dimension: int, device: torch_device):
        super().__init__()
        self.raw_message_dimension = raw_message_dimension
        self.message_dimension = message_dimension
        self.device = device


class MessageFunctionMLP(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int, device: torch_device):
        super().__init__(raw_message_dimension, message_dimension, device)

        self.message_function_net = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        ).to(self.device)

    def forward(self, data):
        computed_return_value = self.message_function_net(data)
        return computed_return_value


class MessageFunctionIdentity(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int, device: torch_device):
        super().__init__(raw_message_dimension, message_dimension, device)
        if raw_message_dimension != message_dimension:
            raise ValueError(f"Identity message dimensions must match: {raw_message_dimension} != {message_dimension}")

    def forward(self, data):
        concat_message = torch_concat(data, dim=-1)

        # returns shape (1, message_dim) (1 row, message dim columns)
        computed_return_value = concat_message.unsqueeze(0)
        return computed_return_value
