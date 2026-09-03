"""Utilities for message aggregator."""

from torch import Tensor as torch_Tensor
from torch import cat as torch_cat
from torch import mean as torch_mean
from torch import nn


class MessageAggregator(nn.Module):
    """
    Base class for all message aggregations
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: list[torch_Tensor]) -> torch_Tensor:
        raise NotImplementedError("MessageAggregator subclasses must implement forward")


class MeanMessageAggregator(MessageAggregator):
    """
    Mean message aggregator
    From messages received as List of torch.Tensor, it creates new aggregated message
    as mean of features received
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: list[torch_Tensor]) -> torch_Tensor:
        if not data:
            raise ValueError("MeanMessageAggregator requires at least one message")
        # here we will 2D tensor
        # shape = (len(data), num_features)
        data_torch = torch_cat(data)
        # mean across rows
        mean = torch_mean(data_torch, dim=0)
        # return shape = (1, num_features)
        computed_return_value = mean.reshape((1, -1))
        return computed_return_value


class LastMessageAggregator(MessageAggregator):
    """
    Last message aggregator
    From messages received as List of torch.Tensor, it returns last message
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: list[torch_Tensor]) -> torch_Tensor:
        if not data:
            raise ValueError("LastMessageAggregator requires at least one message")
        computed_return_value = data[-1]
        return computed_return_value
