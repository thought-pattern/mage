"""Utilities for distance calculator."""

from math import atan2 as math_atan2, cos as math_cos, isfinite as math_isfinite, pi as math_pi, sin as math_sin, sqrt as math_sqrt

KM_MULTIPLIER = 0.001
LATITUDE = "lat"
LONGITUDE = "lng"
MAX_ABS_LATITUDE = 90.0
VALID_METRICS = ["m", "km"]


def calculate_distance_between_points(start: dict[str, float], end: dict[str, float], metrics="m"):
    """
    Returns distance based on the metrics between 2 points.
    :param start: Start node - dictionary with lat and lng
    :param end: End node - dictionary with lat and lng
    :param metrics: m - in metres, km - in kilometres
    :return: float distance
    """

    lat_1 = coordinate_value(start, LATITUDE)
    lng_1 = coordinate_value(start, LONGITUDE)
    lat_2 = coordinate_value(end, LATITUDE)
    lng_2 = coordinate_value(end, LONGITUDE)

    # Longitude wraps around the sphere, but a latitude beyond a pole names no position.
    if abs(lat_1) > MAX_ABS_LATITUDE or abs(lat_2) > MAX_ABS_LATITUDE:
        raise InvalidCoordinatesException("Latitude must lie between -90 and 90 degrees!")

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

    # The haversine term is a squared half-chord bounded by 1; rounding can exceed that bound by one unit in the last
    # place for antipodal points, which would otherwise raise a math domain error on valid coordinates.
    a = min(sin_delta_phi**2 + math_cos(phi_1) * math_cos(phi_2) * (sin_delta_lambda**2), 1.0)
    c = 2 * math_atan2(math_sqrt(a), math_sqrt(1 - a))

    # Distance in metres
    distance = R * c

    if metrics.lower() == "km":
        distance *= KM_MULTIPLIER

    return distance


def coordinate_value(point: dict[str, float], key: str) -> float:
    """
    Admits one coordinate of a point. Absence is decided by the key and the value's type, never by truthiness, so a
    point on the equator or the prime meridian (0.0) is a coordinate. Booleans (the callers' absent-property marker),
    nulls and non-numeric values are refused; numeric strings are parsed; non-finite values are refused.
    """
    value = point.get(key, False)
    if isinstance(value, bool) or value is None:
        raise InvalidCoordinatesException("Latitude/longitude not specified!")
    if not isinstance(value, (int, float, str)):
        raise InvalidCoordinatesException("Latitude/longitude not in numerical format!")

    try:
        coordinate = float(value)
    except ValueError as err:
        raise InvalidCoordinatesException("Latitude/longitude not in numerical format!") from err

    if not math_isfinite(coordinate):
        raise InvalidCoordinatesException("Latitude/longitude must be finite!")

    return coordinate


class InvalidCoordinatesException(Exception):
    pass


class InvalidMetricException(Exception):
    pass
