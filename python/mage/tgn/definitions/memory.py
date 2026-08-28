"""Utilities for memory."""

from typing import Dict

from torch import Tensor as torch_Tensor
from torch import device as torch_device
from torch import float32 as torch_float32
from torch import zeros as torch_zeros


class Memory:
    def __init__(self, memory_dimension: int, device: torch_device):
        self.memory_dimension = memory_dimension
        self.device = device
        self.init_memory()

    def init_memory(self):
        self.memory_container: Dict[int, torch_Tensor] = {}
        self.last_node_update: Dict[int, torch_Tensor] = {}
        return False

    # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second
    # -time-but
    def detach_tensor_grads(self):
        for node_memory in self.memory_container.values():
            if node_memory.grad is not None:
                node_memory.grad.zero_()

        for timestamp in self.last_node_update.values():
            if timestamp.grad is not None:
                timestamp.grad.zero_()
        return False

    def get_node_memory(self, node: int) -> torch_Tensor:
        if node not in self.memory_container:
            self.memory_container[node] = torch_zeros(
                self.memory_dimension,
                dtype=torch_float32,
                device=self.device,
                requires_grad=True,
            )
        _return_value = self.memory_container.get(node, "")
        return _return_value

    def set_node_memory(self, node: int, node_memory: torch_Tensor) -> torch_Tensor:
        self.memory_container[node] = node_memory
        _return_value = self.memory_container.get(node, "")
        return _return_value

    def get_last_node_update(self, node: int) -> torch_Tensor:
        if node not in self.last_node_update:
            self.last_node_update[node] = torch_zeros(
                1, dtype=torch_float32, device=self.device, requires_grad=True
            )
        _return_value = self.last_node_update.get(node, "")
        return _return_value
