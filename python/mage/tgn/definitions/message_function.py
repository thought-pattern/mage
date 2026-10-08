"""Utilities for message function."""

from torch import (
    Tensor as torch_Tensor,
    as_tensor as torch_as_tensor,
    concat as torch_concat,
    device as torch_device,
    float32 as torch_float32,
    nn,
)


class MessageFunction(nn.Module):
    """
    This is base class for Message function implementation
    """

    def __init__(self, raw_message_dimension: int, message_dimension: int, device: torch_device):
        super().__init__()
        self.raw_message_dimension = raw_message_dimension
        self.message_dimension = message_dimension
        self.device = device

    def raw_message(self, data: tuple) -> torch_Tensor:
        """
        Forms the documented raw message by concatenating its parts, for example (s_i, s_j, delta t, e_ij) for an
        interaction or (s_i, t, v_i) for a node event. Tensor parts keep their autograd history; a scalar part such as a
        node event timestamp becomes a one-element tensor.

        :return: raw message of shape (1, raw_message_dimension)
        """
        parts = [torch_as_tensor(part, dtype=torch_float32, device=self.device).reshape(-1) for part in data]
        concat_message = torch_concat(parts)
        if concat_message.shape[0] != self.raw_message_dimension:
            raise ValueError(f"Raw message has {concat_message.shape[0]} values; expected {self.raw_message_dimension}")
        raw_message = concat_message.unsqueeze(0)
        return raw_message


class MessageFunctionMLP(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int, device: torch_device):
        super().__init__(raw_message_dimension, message_dimension, device)

        self.message_function_net = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        ).to(self.device)

    def forward(self, data):
        # shape (1, message_dim)
        computed_return_value = self.message_function_net(self.raw_message(data))
        return computed_return_value


class MessageFunctionIdentity(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int, device: torch_device):
        super().__init__(raw_message_dimension, message_dimension, device)
        if raw_message_dimension != message_dimension:
            raise ValueError(f"Identity message dimensions must match: {raw_message_dimension} != {message_dimension}")

    def forward(self, data):
        # returns shape (1, message_dim) (1 row, message dim columns)
        computed_return_value = self.raw_message(data)
        return computed_return_value
