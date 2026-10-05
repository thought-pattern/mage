"""
Vehicle routing solver contract. A solved route is a list of path dictionaries {"from_vertex": int, "to_vertex": int},
one per chosen edge, whose endpoints are positions in the distance matrix the solver was given.
"""

from abc import ABC, abstractmethod


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
        Extract the chosen route edges from the solved problem as path dictionaries.
        """
        ...


class InvalidDepotException(Exception):
    pass
