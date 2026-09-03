"""Utilities for convergence callback."""

from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.iteration_callbacks.iteration_callback import (
    IterationCallback,
)
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate


class ConvergenceCallback(IterationCallback):
    """
    A class that represents Convergence Callback. This iteration
    callback after each iteration checks whether the algorithm has
    found a better solution than the existing one. If the algorithm
    did not find a better solution, the number of iterations in which
    a better solution was not found increases. When that number of
    iterations reaches the default number, defined actions are called
    and the iterations counter is set to zero.
    """

    def __init__(self):
        self.internal_iteration = 0
        self.best_solution_error = float("inf")
        super().__init__()

    @validate(Parameter.ERROR, Parameter.CONVERGENCE_CALLBACK_TOLERANCE)
    def update(self, graph: Graph, population: Population, parameters: dict):
        error = param_value(graph, parameters, Parameter.ERROR)
        convergence_callback_tolerance = param_value(graph, parameters, Parameter.CONVERGENCE_CALLBACK_TOLERANCE)

        if self.best_solution_error == float("inf"):
            self.internal_iteration = 1
            self.best_solution_error = population.min_error(error.individual_err)
        else:
            self.internal_iteration += 1
            if population.min_error(error.individual_err) < self.best_solution_error:
                self.best_solution_error = population.min_error(error.individual_err)
                self.internal_iteration = 0

        if self.internal_iteration == convergence_callback_tolerance:
            self.convergence_detected(graph, population, parameters)
        return False

    def end(self, graph: Graph, population: Population, parameters: dict):
        return False

    @validate(Parameter.ERROR, Parameter.CONVERGENCE_CALLBACK_ACTIONS)
    def convergence_detected(self, graph: Graph, population: Population, parameters: dict):
        error = param_value(graph, parameters, Parameter.ERROR)
        convergence_callback_actions = param_value(graph, parameters, Parameter.CONVERGENCE_CALLBACK_ACTIONS)

        for action in convergence_callback_actions:
            action.execute(graph, population, parameters)

        self.internal_iteration = 0
        self.best_solution_error = population.min_error(error.individual_err)
        return False
