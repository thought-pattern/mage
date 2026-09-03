"""
VRP Path is an edge from a starting to ending node
"""

from abc import ABC, abstractmethod


class VRPPath(tuple):
    __slots__ = ()
    fields = ("from_vertex", "to_vertex")
    __match_args__ = fields

    def __new__(cls, from_vertex, to_vertex):
        values = (from_vertex, to_vertex)
        result = tuple.__new__(cls, values)
        return result

    def __getnewargs__(self):
        result = tuple(self)
        return result

    def internal_replace(self, **changes):
        unknown_fields = set(changes) - set(self.fields)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"unexpected record fields: {names}")
        values = tuple(changes.get(field, tuple.__getitem__(self, index)) for index, field in enumerate(self.fields))
        result = type(self)(*values)
        return result

    def as_dict(self):
        result = {field: tuple.__getitem__(self, index) for index, field in enumerate(self.fields)}
        return result

    def tree_flatten(self):
        children = tuple(self)
        metadata = ()
        result = (children, metadata)
        return result

    @classmethod
    def tree_unflatten(cls, metadata, children):
        del metadata
        result = cls(*children)
        return result

    def __repr__(self):
        pairs = ", ".join(f"{field}={tuple.__getitem__(self, index)!r}" for index, field in enumerate(self.fields))
        result = f"{type(self).__name__}({pairs})"
        return result

    @property
    def from_vertex(self):
        result = tuple.__getitem__(self, 0)
        return result

    @property
    def to_vertex(self):
        result = tuple.__getitem__(self, 1)
        return result


class VRPResult:
    """
    The VRP Result consists of multiple VRP paths.
    """

    def __init__(self, vrp_paths: list[VRPPath]):
        self.vrp_paths = vrp_paths


class VRPSolver(ABC):
    """
    VRP Solver solves the VRP problem and can extract results to desired hook.
    """

    @abstractmethod
    def solve(self):
        """
        Implementation method.
        """
        ...

    @abstractmethod
    def get_result(self):
        """
        Extract results from solved problem.
        """
        ...


class InvalidDepotException(Exception):
    pass
