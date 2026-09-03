"""External parameter normalization for graph-coloring procedures."""

from mage.graph_coloring_module.algorithms.greedy.LDO import LDO
from mage.graph_coloring_module.algorithms.greedy.random import Random
from mage.graph_coloring_module.algorithms.greedy.SDO import SDO
from mage.graph_coloring_module.algorithms.meta_heuristics.quantum_annealing import QA
from mage.graph_coloring_module.error_functions.conflict_error import ConflictError
from mage.graph_coloring_module.exceptions import IncorrectParametersException
from mage.graph_coloring_module.iteration_callbacks.callback_actions.simple_tunneling import (
    SimpleTunneling,
)
from mage.graph_coloring_module.iteration_callbacks.convergence_callback import (
    ConvergenceCallback,
)
from mage.graph_coloring_module.operators.mutations.MIS_mutation import MISMutation
from mage.graph_coloring_module.operators.mutations.multiple_mutation import (
    MultipleMutation,
)
from mage.graph_coloring_module.operators.mutations.random_mutation import RandomMutation
from mage.graph_coloring_module.operators.mutations.simple_mutation import SimpleMutation
from mage.graph_coloring_module.parameters import Parameter


def normalize_parameters(parameters: dict) -> dict:
    """Validate external values and bind finite names to concrete behaviors."""

    if not isinstance(parameters, dict):
        raise TypeError("graph-coloring parameters must be a dictionary")

    algorithm_names = {
        "QA": QA(),
        "LDO": LDO(),
        "SDO": SDO(),
        "Random": Random(),
    }
    initialization_names = {
        "LDO": LDO(),
        "SDO": SDO(),
        "Random": Random(),
    }
    error_names = {"ConflictError": ConflictError()}
    callback_names = {"ConvergenceCallback": ConvergenceCallback()}
    mutation_names = {
        "MISMutation": MISMutation(),
        "MultipleMutation": MultipleMutation(),
        "RandomMutation": RandomMutation(),
        "SimpleMutation": SimpleMutation(),
    }
    action_names = {"SimpleTunneling": SimpleTunneling()}

    algorithm_setting = parameters.get(Parameter.ALGORITHM.value, "QA")
    if isinstance(algorithm_setting, str):
        algorithm = algorithm_names.get(algorithm_setting, False)
    else:
        algorithm = algorithm_setting
    if not callable(getattr(algorithm, "run", False)):
        raise IncorrectParametersException(f"Unknown graph-coloring algorithm: {algorithm_setting!r}")

    initialization_settings = parameters.get(Parameter.INIT_ALGORITHMS.value, ["SDO", "LDO"])
    if not isinstance(initialization_settings, list):
        raise IncorrectParametersException("init_algorithms must be a list")
    initialization_algorithms = []
    for setting in initialization_settings:
        if isinstance(setting, str):
            initialization_algorithm = initialization_names.get(setting, False)
        else:
            initialization_algorithm = setting
        if not callable(getattr(initialization_algorithm, "run", False)):
            raise IncorrectParametersException(f"Unknown initialization algorithm: {setting!r}")
        initialization_algorithms.append(initialization_algorithm)

    error_setting = parameters.get(Parameter.ERROR.value, "ConflictError")
    if isinstance(error_setting, str):
        error = error_names.get(error_setting, False)
    else:
        error = error_setting
    if not callable(getattr(error, "individual_err", False)) or not callable(getattr(error, "population_err", False)):
        raise IncorrectParametersException(f"Unknown graph-coloring error: {error_setting!r}")

    callback_settings = parameters.get(Parameter.ITERATION_CALLBACKS.value, [])
    if not isinstance(callback_settings, list):
        raise IncorrectParametersException("iteration_callbacks must be a list")
    callbacks = []
    for setting in callback_settings:
        if isinstance(setting, str):
            callback = callback_names.get(setting, False)
        else:
            callback = setting
        if not callable(getattr(callback, "update", False)) or not callable(getattr(callback, "end", False)):
            raise IncorrectParametersException(f"Unknown iteration callback: {setting!r}")
        callbacks.append(callback)

    mutation_setting = parameters.get(Parameter.MUTATION.value, "SimpleMutation")
    if isinstance(mutation_setting, str):
        mutation = mutation_names.get(mutation_setting, False)
    else:
        mutation = mutation_setting
    if not callable(getattr(mutation, "mutate", False)):
        raise IncorrectParametersException(f"Unknown graph-coloring mutation: {mutation_setting!r}")

    tunneling_mutation_setting = parameters.get(Parameter.SIMPLE_TUNNELING_MUTATION.value, "MultipleMutation")
    if isinstance(tunneling_mutation_setting, str):
        tunneling_mutation = mutation_names.get(tunneling_mutation_setting, False)
    else:
        tunneling_mutation = tunneling_mutation_setting
    if not callable(getattr(tunneling_mutation, "mutate", False)):
        raise IncorrectParametersException(f"Unknown tunneling mutation: {tunneling_mutation_setting!r}")

    action_settings = parameters.get(Parameter.CONVERGENCE_CALLBACK_ACTIONS.value, ["SimpleTunneling"])
    if not isinstance(action_settings, list):
        raise IncorrectParametersException("convergence_callback_actions must be a list")
    actions = []
    for setting in action_settings:
        if isinstance(setting, str):
            action = action_names.get(setting, False)
        else:
            action = setting
        if not callable(getattr(action, "execute", False)):
            raise IncorrectParametersException(f"Unknown convergence action: {setting!r}")
        actions.append(action)

    normalized = {
        Parameter.ALGORITHM: algorithm,
        Parameter.NO_OF_COLORS: parameters.get(Parameter.NO_OF_COLORS.value, 10),
        Parameter.NO_OF_PROCESSES: parameters.get(Parameter.NO_OF_PROCESSES.value, 1),
        Parameter.POPULATION_SIZE: parameters.get(Parameter.POPULATION_SIZE.value, 15),
        Parameter.INIT_ALGORITHMS: initialization_algorithms,
        Parameter.ERROR: error,
        Parameter.MAX_ITERATIONS: parameters.get(Parameter.MAX_ITERATIONS.value, 10),
        Parameter.ITERATION_CALLBACKS: callbacks,
        Parameter.COMMUNICATION_DALAY: parameters.get(Parameter.COMMUNICATION_DALAY.value, 10),
        Parameter.LOGGING_DELAY: parameters.get(Parameter.LOGGING_DELAY.value, 10),
        Parameter.QA_TEMPERATURE: parameters.get(Parameter.QA_TEMPERATURE.value, 0.035),
        Parameter.QA_MAX_STEPS: parameters.get(Parameter.QA_MAX_STEPS.value, 10),
        Parameter.CONFLICT_ERR_ALPHA: parameters.get(Parameter.CONFLICT_ERR_ALPHA.value, 0.1),
        Parameter.CONFLICT_ERR_BETA: parameters.get(Parameter.CONFLICT_ERR_BETA.value, 0.001),
        Parameter.MUTATION: mutation,
        Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES: parameters.get(Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES.value, 2),
        Parameter.RANDOM_MUTATION_PROBABILITY: parameters.get(Parameter.RANDOM_MUTATION_PROBABILITY.value, 0.1),
        Parameter.SIMPLE_TUNNELING_MUTATION: tunneling_mutation,
        Parameter.SIMPLE_TUNNELING_PROBABILITY: parameters.get(Parameter.SIMPLE_TUNNELING_PROBABILITY.value, 0.5),
        Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION: parameters.get(Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION.value, 2),
        Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS: parameters.get(Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS.value, 25),
        Parameter.CONVERGENCE_CALLBACK_TOLERANCE: parameters.get(Parameter.CONVERGENCE_CALLBACK_TOLERANCE.value, 500),
        Parameter.CONVERGENCE_CALLBACK_ACTIONS: actions,
    }

    positive_integer_parameters = (
        Parameter.NO_OF_COLORS,
        Parameter.NO_OF_PROCESSES,
        Parameter.POPULATION_SIZE,
        Parameter.MAX_ITERATIONS,
        Parameter.COMMUNICATION_DALAY,
        Parameter.LOGGING_DELAY,
        Parameter.QA_MAX_STEPS,
        Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES,
        Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS,
        Parameter.CONVERGENCE_CALLBACK_TOLERANCE,
    )
    for parameter in positive_integer_parameters:
        value = normalized.get(parameter, False)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise IncorrectParametersException(f"{parameter.value} must be a positive integer")

    positive_number_parameters = (Parameter.QA_TEMPERATURE,)
    for parameter in positive_number_parameters:
        value = normalized.get(parameter, False)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise IncorrectParametersException(f"{parameter.value} must be positive")

    probability_parameters = (
        Parameter.RANDOM_MUTATION_PROBABILITY,
        Parameter.SIMPLE_TUNNELING_PROBABILITY,
    )
    for parameter in probability_parameters:
        value = normalized.get(parameter, False)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise IncorrectParametersException(f"{parameter.value} must be between zero and one")

    numeric_parameters = (
        Parameter.CONFLICT_ERR_ALPHA,
        Parameter.CONFLICT_ERR_BETA,
        Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION,
    )
    for parameter in numeric_parameters:
        value = normalized.get(parameter, False)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise IncorrectParametersException(f"{parameter.value} must be numeric")

    return normalized
