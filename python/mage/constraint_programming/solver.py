"""Utilities for solver."""

from abc import ABC as abc_ABC, abstractmethod as abc_abstractmethod
from sys import stderr as sys_stderr, version as sys_version

try:
    from gekko import GEKKO
except ImportError:
    sys_stderr.write(f"NOTE: Please install gekko in order to be able to use set-cover solver. Using Python: {sys_version}")
    raise


class MatchingProblem:
    """
    Definition for matching problem of set cover
    """


class GreedyMatchingProblem(MatchingProblem):
    """
    Matching problem to be used with greedy solving of set cover.
    """

    def __init__(self, elements, containing_sets, elements_by_sets):
        self.elements = elements
        self.containing_sets = containing_sets
        self.elements_by_sets = elements_by_sets


class GekkoMatchingProblem(MatchingProblem):
    """
    Matching problem to be used with gekko constraint programming solving of set cover.
    """

    def __init__(self, containing_sets, sets_by_elements):
        self.containing_sets = containing_sets
        self.sets_by_elements = sets_by_elements


class MatchingProblemSolver(abc_ABC):
    """
    Solver of set cover matching problem
    """

    @abc_abstractmethod
    def solve(self, matching_problem: MatchingProblem):
        """
        Solves the matching problem and returns the set indices
        :param matching_problem: matching problem
        :return: set indices
        """
        ...


class GekkoMPSolver(MatchingProblemSolver):
    """
    Solver of set cover with gekko constraint programming
    """

    def solve(self, matching_problem: GekkoMatchingProblem):
        """
        Solves the matching problem and returns the set indices
        :param matching_problem: matching problem
        :return: set indices
        """

        m = GEKKO(remote=False)
        m.options.SOLVER = 1
        containing_const = m.Const(1, name="const")

        set_list = list(matching_problem.containing_sets)
        vars = [m.Var(lb=0, ub=1, integer=True, name=GekkoMPSolver.get_variable_name(i)) for i in range(len(set_list))]

        set_ordinal_map = {value: i for i, value in enumerate(set_list)}

        for element, containing_sets in matching_problem.sets_by_elements.items():
            contained_set_variables = []

            for contained_set in containing_sets:
                if contained_set not in set_ordinal_map:
                    raise KeyError(f"unknown containing set {contained_set!r}")
                contained_set_variables.append(vars[set_ordinal_map.get(contained_set, 0)])

            if not contained_set_variables:
                raise ValueError(f"element {element!r} is not contained by any set")
            contained_sets_equation = contained_set_variables[0]
            for contained_set_variable in contained_set_variables[1:]:
                contained_sets_equation += contained_set_variable
            m.Equation(equation=contained_sets_equation >= containing_const)

        m.Obj(sum(vars))
        m.solve()

        resulting_sets = []
        for idx, var in enumerate(vars):
            if var.value[0] == 1.0:
                resulting_sets.append(set_list[idx])

        return resulting_sets

    @staticmethod
    def get_variable_name(set_no):
        """
        Returns unique variable name based on the set id
        :param set_no: set id
        :return: set variable name
        """

        computed_return_value = f"containing_set_{set_no}"
        return computed_return_value


class GreedyMPSolver(MatchingProblemSolver):
    """
    Solver of set cover with the classic greedy method: repeatedly pick the set covering the most still-uncovered
    elements. This is a heuristic with the H(n) <= ln(n) + 1 approximation bound (n = largest set size), not an exact
    minimum; GekkoMPSolver is the exact solver.
    """

    def solve(self, matching_problem: GreedyMatchingProblem):
        """
        Solves the matching problem and returns the set indices
        :param matching_problem: matching problem
        :return: set indices
        """

        uncovered = set(matching_problem.elements)
        # Sorted candidates make ties explicit and deterministic: among equally useful sets the lowest id is taken.
        candidate_sets = sorted(matching_problem.containing_sets)

        picked_sets = []
        while uncovered and candidate_sets:
            picked_set = max(
                candidate_sets,
                key=lambda candidate: len(uncovered & matching_problem.elements_by_sets.get(candidate, set())),
            )
            newly_covered = uncovered & matching_problem.elements_by_sets.get(picked_set, set())
            if not newly_covered:
                break

            picked_sets.append(picked_set)
            candidate_sets.remove(picked_set)
            uncovered -= newly_covered

        if uncovered:
            raise ValueError(f"elements {sorted(uncovered)!r} are not contained by any set")

        return picked_sets
