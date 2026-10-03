"""Utilities for algorithm."""

from abc import ABC, abstractmethod

from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.graph import Graph


def no_host_abort() -> bool:
    """Abort check for runs outside a Memgraph query: no host can cancel them, so it never raises."""
    return False


class Algorithm(ABC):
    """An abstract class that represents an algorithm."""

    @abstractmethod
    def run(self, graph: Graph, parameters: dict, abort_check=no_host_abort) -> Individual:
        """Runs the algorithm and returns the best individual. `abort_check` is the host's cancellation probe
        (Memgraph's ProcCtx.check_must_abort), which raises when the host aborts the query; work that can run long
        polls it so that the cancellation propagates out of the procedure."""
        ...
