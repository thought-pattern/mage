"""Utilities for math functions."""

from numpy import array as np_array


def normalize(an_array: list[float]) -> list[float]:
    computed_return_value = np_array(an_array) / sum(an_array)
    result_list = computed_return_value.tolist()
    return result_list
