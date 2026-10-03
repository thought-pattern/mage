"""Utilities for validation."""

from inspect import Parameter as inspect_Parameter
from inspect import signature as inspect_signature

from mage.graph_coloring_module.exceptions import MissingParametersException

PARAMETERS_ARGUMENT = "parameters"


def validate(*params_name):
    def check_accepts(f):
        # The dictionary is located through the decorated function's declared `parameters` argument, so positional,
        # keyword and declared-default invocations are admitted by the same rule.
        declared_arguments = inspect_signature(f).parameters
        declared_parameters = declared_arguments.get(PARAMETERS_ARGUMENT, False)
        if not isinstance(declared_parameters, inspect_Parameter):
            raise TypeError(f"validated function {f.__name__} must declare a '{PARAMETERS_ARGUMENT}' argument")
        parameters_position = list(declared_arguments).index(PARAMETERS_ARGUMENT)
        declared_default = declared_parameters.default if isinstance(declared_parameters.default, dict) else {}

        def validated_call(*args, **kwds):
            if PARAMETERS_ARGUMENT in kwds:
                parameters = kwds.get(PARAMETERS_ARGUMENT, {})
            elif len(args) > parameters_position:
                parameters = args[parameters_position]
            else:
                parameters = declared_default
            for param in params_name:
                if not isinstance(parameters, dict) or not parameters:
                    raise MissingParametersException("Missing parameters in function {}".format(f.__name__))
                if param not in parameters:
                    raise MissingParametersException("Missing parameter {} in function {}".format(param, f.__name__))
            computed_return_value = f(*args, **kwds)
            return computed_return_value

        validated_call.__name__ = f.__name__
        return validated_call

    return check_accepts
