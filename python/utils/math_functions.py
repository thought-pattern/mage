"""Utilities for math functions."""

from typing import List

from numpy import array as np_array


def normalize(an_array: List[float]) -> List[float]:
    _return_value = np_array(an_array) / sum(an_array)
    return _return_value
