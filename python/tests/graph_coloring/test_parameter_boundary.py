"""Behavior tests for the external graph-coloring parameter boundary."""

from pytest import raises

from mage.graph_coloring_module import ConflictError, Parameter, QA
from mage.graph_coloring_module.exceptions import IncorrectParametersException
from mage.graph_coloring_module.parameter_boundary import normalize_parameters


def test_named_parameters_bind_category_owned_behaviors():
    parameters = normalize_parameters(
        {
            "algorithm": "QA",
            "error": "ConflictError",
            "iteration_callbacks": ["ConvergenceCallback"],
            "mutation": "SimpleMutation",
            "convergence_callback_actions": ["SimpleTunneling"],
        }
    )

    assert isinstance(parameters.get(Parameter.ALGORITHM, False), QA)
    assert isinstance(parameters.get(Parameter.ERROR, False), ConflictError)
    assert callable(getattr(parameters.get(Parameter.MUTATION, False), "mutate", False))
    assert callable(getattr(parameters.get(Parameter.ITERATION_CALLBACKS, [False])[0], "update", False))
    assert callable(getattr(parameters.get(Parameter.CONVERGENCE_CALLBACK_ACTIONS, [False])[0], "execute", False))


def test_parameter_category_rejects_public_but_wrong_class_name():
    with raises(IncorrectParametersException, match="algorithm"):
        normalize_parameters({"algorithm": "ConflictError"})


def test_invalid_process_count_is_rejected_at_boundary():
    with raises(IncorrectParametersException, match="positive integer"):
        normalize_parameters({"no_of_processes": 0})
