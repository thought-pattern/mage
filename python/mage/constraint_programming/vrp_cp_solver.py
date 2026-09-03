"""Utilities for vrp cp solver."""

from abc import ABC, abstractmethod

from gekko import GEKKO
from numpy import ndarray as np_ndarray

from mage.geography import InvalidDepotException, VRPPath, VRPResult, VRPSolver


class VRPConstraintProgrammingSolver(VRPSolver):
    """
    This constraint solver solves the Vehicle Routing Problem with constraint programming using GEKKO.
    """

    SOURCE_INDEX = -1
    SINK_INDEX = -2

    def __init__(self, no_vehicles: int, distance_matrix: np_ndarray, depot_index: int):
        if depot_index < 0 or depot_index >= len(distance_matrix):
            raise InvalidDepotException("Depot index outside the range of locations!")

        self.internal_model = GEKKO(remote=False)

        self.no_vehicles = no_vehicles
        self.distance_matrix = distance_matrix
        self.depot_index = depot_index

        self.edge_chosen_vars = dict()
        self.internal_time_vars = dict()
        self.location_node_ids = [x for x in range(len(distance_matrix)) if x != self.depot_index]

        self.internal_constraints: list[VRPConstraint] = [
            TimeIncreasesWithPassingFromOneNodeToAnotherConstraint(
                self.internal_model,
                self.edge_chosen_vars,
                self.internal_time_vars,
                self.distance_matrix,
            ),
            No3NodeCyclesConstraint(
                self.internal_model,
                self.edge_chosen_vars,
                self.location_node_ids,
            ),
            StartInSourceNodeConstraint(
                self.internal_model,
                self.edge_chosen_vars,
                self.location_node_ids,
                self.no_vehicles,
                self.SOURCE_INDEX,
            ),
            EndInSinkNodeConstraint(
                self.internal_model,
                self.edge_chosen_vars,
                self.location_node_ids,
                self.no_vehicles,
                self.SINK_INDEX,
            ),
            MaximumEdgesActivatedConstraint(
                self.internal_model,
                self.edge_chosen_vars,
                self.location_node_ids,
                self.no_vehicles,
            ),
            NoBacktrackingConstraint(self.internal_model, self.edge_chosen_vars),
        ]

        self.initialize()
        self.add_constraints()
        self.add_objective()
        self.add_options()

    def solve(self):
        self.internal_model.solve()
        return False

    def get_result(self) -> VRPResult:
        computed_return_value = VRPResult(
            [
                VRPPath(
                    key[0] if key[0] >= 0 else self.depot_index,
                    key[1] if key[1] >= 0 else self.depot_index,
                )
                for key, var in self.edge_chosen_vars.items()
                if int(var.value[0]) == 1
            ]
        )
        return computed_return_value

    def get_distance(self, edge: tuple[int, int]) -> float:
        node_from, node_to = edge

        if any(node in [self.SOURCE_INDEX, self.SINK_INDEX] for node in [node_from, node_to]):
            return 0.0

        computed_return_value = self.distance_matrix[node_from][node_to]
        return computed_return_value

    def initialize(self):
        for node_index in range(len(self.distance_matrix)):
            if node_index in self.location_node_ids:
                self.initialize_location_node(node_index)
        return False

    def initialize_location_node(self, node_index: int):
        self.internal_time_vars[node_index] = self.internal_model.Var(value=0, lb=0, integer=False)

        # Initialize starting point and sinking point for every vehicle
        self.add_variable((self.SOURCE_INDEX, node_index))
        self.add_variable((node_index, self.SINK_INDEX))

        # For every node, draw lengths from and to it, with duration of edges
        out_vars = self.add_adjacent_output_edge_variables(node_index)
        in_vars = self.add_adjacent_input_edge_variables(node_index)

        # Either it was a beginning node, or a vehicle has visited it in the drive.
        if len(out_vars) > 0:
            self.internal_model.Equation(self.edge_chosen_vars.get((node_index, self.SINK_INDEX), 0.0) + sum(out_vars) == 1)

        if len(in_vars) > 0:
            self.internal_model.Equation(self.edge_chosen_vars.get((self.SOURCE_INDEX, node_index), 0.0) + sum(in_vars) == 1)
        return False

    def add_adjacent_output_edge_variables(self, node_index: int):
        edges_vars = []

        for adjacent_node in range(len(self.distance_matrix)):
            if adjacent_node == self.depot_index:
                continue

            edge = (node_index, adjacent_node)
            var = self.add_variable(edge)
            edges_vars.append(var)

        return edges_vars

    def add_adjacent_input_edge_variables(self, node_index: int):
        edges_vars = []

        for adjacent_node in range(len(self.distance_matrix)):
            if adjacent_node == self.depot_index:
                continue

            edge = (adjacent_node, node_index)
            var = self.add_variable(edge)
            edges_vars.append(var)

        return edges_vars

    def add_variable(self, edge: tuple[int, int]):
        var = self.edge_chosen_vars.get(edge, False)

        if var is False:
            var = self.internal_model.Var(value=0, lb=0, ub=1, integer=True)
            self.edge_chosen_vars[edge] = var

        return var

    def add_constraints(self):
        """
        Add global constraints to the solver.
        """
        for constraint in self.internal_constraints:
            constraint.apply_constraint()
        return False

    def add_objective(self):
        intermediate_sum = 0
        for edge, variable in self.edge_chosen_vars.items():
            duration = self.get_distance(edge)
            intermediate_sum += self.internal_model.Intermediate(duration * variable)

        self.internal_model.Obj(intermediate_sum)
        return False

    def add_options(self):
        # The SOLVER option specifies the type of solver that solves the
        # VRP problem. More on solver options and other parameters can be found on
        # https://gekko.readthedocs.io/en/latest/global.html
        self.internal_model.options.SOLVER = 1
        return False


class VRPConstraint(ABC):
    def __init__(self, model: GEKKO):
        self.internal_model = model

    @abstractmethod
    def apply_constraint(self): ...


class TimeIncreasesWithPassingFromOneNodeToAnotherConstraint(VRPConstraint):
    """
    Allow progression in time when passing from one node to another.
    """

    def __init__(self, model: GEKKO, variables, time_vars, distance_matrix: np_ndarray):
        super().__init__(model)

        self.internal_variables = variables
        self.time_variables = time_vars
        self.internal_distance_matrix = distance_matrix

    def apply_constraint(self):
        for edge in self.internal_variables:
            (from_node, to_node) = edge
            if from_node < 0 or to_node < 0:
                continue

            self.internal_model.Equation(
                (self.time_variables[from_node] + self.internal_distance_matrix[from_node][to_node]) * self.internal_variables[edge]
                <= self.time_variables[to_node]
            )
        return False


class No3NodeCyclesConstraint(VRPConstraint):
    """
    Do not allow 3 node loops
    """

    def __init__(self, model: GEKKO, variables, node_ids: list[int]):
        super().__init__(model)

        self.internal_variables = variables
        self.internal_node_ids = node_ids

    def apply_constraint(self):
        """
        Do not allow 3 node loops
        """
        for a in self.internal_node_ids:
            for b in self.internal_node_ids:
                if a == b:
                    continue
                for c in self.internal_node_ids:
                    if c == a or c == b:
                        continue
                    self.internal_model.Equation(
                        self.internal_variables[(a, b)] + self.internal_variables[(b, c)] + self.internal_variables[(c, a)] <= 2
                    )
        return False


class StartInSourceNodeConstraint(VRPConstraint):
    """
    Whatever the source node is, all of the vehicles must be found in it at some point.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
        node_ids: list[int],
        no_vehicles: int,
        source_id: int,
    ):
        super().__init__(model)

        self.internal_variables = variables
        self.internal_node_ids = node_ids
        self.internal_no_vehicles = no_vehicles
        self.internal_source_id = source_id

    def apply_constraint(self):
        self.internal_model.Equation(
            sum(self.internal_variables[(self.internal_source_id, n)] for n in self.internal_node_ids) == self.internal_no_vehicles
        )
        return False


class EndInSinkNodeConstraint(VRPConstraint):
    """
    Whatever the sink node is, all of the vehicles must be found in it at some point.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
        node_ids: list[int],
        no_vehicles: int,
        sink_id: int,
    ):
        super().__init__(model)

        self.internal_variables = variables
        self.internal_node_ids = node_ids
        self.internal_no_vehicles = no_vehicles
        self.internal_sink_id = sink_id

    def apply_constraint(self):
        self.internal_model.Equation(
            sum(self.internal_variables[(n, self.internal_sink_id)] for n in self.internal_node_ids) == self.internal_no_vehicles
        )
        return False


class MaximumEdgesActivatedConstraint(VRPConstraint):
    """
    Add total number of paths (edges) that needs to be present.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
        node_ids: list[int],
        no_vehicles: int,
    ):
        super().__init__(model)

        self.internal_variables = variables
        self.internal_node_ids = node_ids
        self.internal_no_vehicles = no_vehicles

    def apply_constraint(self):
        self.internal_model.Equation(
            sum(self.internal_variables.values()) == len(self.internal_node_ids) + self.internal_no_vehicles
        )
        return False


class NoBacktrackingConstraint(VRPConstraint):
    """
    Add no backtracking from one node to another.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
    ):
        super().__init__(model)
        self.internal_variables = variables

    def apply_constraint(self):
        for edge in self.internal_variables:
            (from_node, to_node) = edge
            if from_node < 0 or to_node < 0:
                continue

            self.internal_model.Equation(
                self.internal_variables[(from_node, to_node)] + self.internal_variables[(to_node, from_node)] <= 1
            )
        return False
