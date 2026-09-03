"""Utilities for parameters utils."""

from inspect import isfunction as inspect_isfunction

from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.parameters import Parameter


def param_value(graph: Graph, parameters: dict, param: Parameter):
    if not isinstance(parameters, dict):
        raise TypeError("graph-coloring parameters must be a dictionary")
    if not isinstance(param, Parameter):
        raise TypeError("graph-coloring parameter keys must be Parameter values")

    if param not in parameters:
        raise KeyError(f"missing graph-coloring parameter {param.value}")
    value = parameters.get(param, False)
    if value is False:
        raise ValueError(f"graph-coloring parameter {param.value} cannot be disabled")

    if inspect_isfunction(value):
        value = value(graph)

    return value
