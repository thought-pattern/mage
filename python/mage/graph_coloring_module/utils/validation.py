"""Utilities for validation."""

from mage.graph_coloring_module.exceptions import MissingParametersException


def validate(*params_name):
    def check_accepts(f):
        def new_f(*args, **kwds):
            parameters = {}
            for arg in args:
                if isinstance(arg, dict):
                    parameters = arg
            for param in params_name:
                if not parameters:
                    raise MissingParametersException(
                        "Missing parameters in function {}".format(f.__name__)
                    )
                if param not in parameters:
                    raise MissingParametersException(
                        "Missing parameter {} in function {}".format(param, f.__name__)
                    )
            _return_value = f(*args, **kwds)
            return _return_value

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts
