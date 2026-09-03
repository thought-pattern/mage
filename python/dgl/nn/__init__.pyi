"""Static contract for DGL neural-network layers."""

from typing import Any as Dynamic
from torch import nn as torch_nn

class HeteroGraphConv(torch_nn.Module):
    def __init__(self, *args: Dynamic, **kwargs: Dynamic) -> None: ...
    def __call__(self, *args: Dynamic, **kwargs: Dynamic) -> Dynamic: ...
    def __getattr__(self, name: str) -> Dynamic: ...

pytorch: Dynamic
