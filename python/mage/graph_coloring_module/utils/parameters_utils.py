"""Utilities for parameters utils."""

from types import FunctionType as types_FunctionType
from typing import Any, Dict

from mage.graph_coloring_module.graph import Graph


def param_value(
    graph: Graph, parameters: Dict[str, Any], param: str, initial_value: Any = False
) -> Any:
    if initial_value is None:
        initial_value = False
    if parameters is None:
        if initial_value is False:
            return False
        return initial_value

    param = parameters.get(param, False)

    if param is False:
        if initial_value is False:
            return False
        return initial_value

    if isinstance(param, types_FunctionType):
        param = param(graph)

    return param
