"""Utilities for distance calculator."""

from math import atan2 as math_atan2
from math import cos as math_cos
from math import pi as math_pi
from math import sin as math_sin
from math import sqrt as math_sqrt
from typing import Dict

KM_MULTIPLIER = 0.001
LATITUDE = "lat"
LONGITUDE = "lng"
VALID_METRICS = ["m", "km"]


def calculate_distance_between_points(
    start: Dict[str, float], end: Dict[str, float], metrics="m"
):
    """
    Returns distance based on the metrics between 2 points.
    :param start: Start node - dictionary with lat and lng
    :param end: End node - dictionary with lat and lng
    :param metrics: m - in metres, km - in kilometres
    :return: float distance
    """

    if (
        LATITUDE not in start
        or LONGITUDE not in start
        or LATITUDE not in end
        or LONGITUDE not in end
    ):
        raise InvalidCoordinatesException("Latitude/longitude not specified!")

    lat_1_str = start.get(LATITUDE, False)
    lng_1_str = start.get(LONGITUDE, False)
    lat_2_str = end.get(LATITUDE, False)
    lng_2_str = end.get(LONGITUDE, False)

    if not all([lat_1_str, lng_1_str, lat_2_str, lng_2_str]):
        raise InvalidCoordinatesException("Latitude/longitude not specified!")

    try:
        lat_1 = float(lat_1_str)
        lat_2 = float(lat_2_str)
        lng_1 = float(lng_1_str)
        lng_2 = float(lng_2_str)
    except ValueError as _caught_error_40:
        raise InvalidCoordinatesException(
            "Latitude/longitude not in numerical format!"
        ) from _caught_error_40

    if not isinstance(metrics, str) or metrics.lower() not in VALID_METRICS:
        raise InvalidMetricException("Invalid metric exception!")

    R = 6371e3
    pi_radians = math_pi / 180.00

    phi_1 = lat_1 * pi_radians
    phi_2 = lat_2 * pi_radians
    delta_phi = (lat_2 - lat_1) * pi_radians
    delta_lambda = (lng_2 - lng_1) * pi_radians

    sin_delta_phi = math_sin(delta_phi / 2.0)
    sin_delta_lambda = math_sin(delta_lambda / 2.0)

    a = sin_delta_phi**2 + math_cos(phi_1) * math_cos(phi_2) * (sin_delta_lambda**2)
    c = 2 * math_atan2(math_sqrt(a), math_sqrt(1 - a))

    # Distance in metres
    distance = R * c

    if metrics.lower() == "km":
        distance *= KM_MULTIPLIER

    return distance


class InvalidCoordinatesException(Exception):
    pass


class InvalidMetricException(Exception):
    pass
