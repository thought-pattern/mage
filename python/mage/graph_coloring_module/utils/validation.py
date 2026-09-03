"""Utilities for validation."""

from mage.graph_coloring_module.exceptions import MissingParametersException


def validate(*params_name):
    def check_accepts(f):
        def validated_call(*args, **kwds):
            parameters = {}
            for arg in args:
                if isinstance(arg, dict):
                    parameters = arg
            for param in params_name:
                if not parameters:
                    raise MissingParametersException("Missing parameters in function {}".format(f.__name__))
                if param not in parameters:
                    raise MissingParametersException("Missing parameter {} in function {}".format(param, f.__name__))
            computed_return_value = f(*args, **kwds)
            return computed_return_value

        validated_call.__name__ = f.__name__
        return validated_call

    return check_accepts
