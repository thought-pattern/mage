"""Public API for the geography package."""

from mage.geography.distance_calculator import (
    LATITUDE,  # noqa: F401
    LONGITUDE,  # noqa: F401
    InvalidCoordinatesException,  # noqa: F401
    InvalidMetricException,  # noqa: F401
    calculate_distance_between_points,  # noqa: F401
)
from mage.geography.travelling_salesman import (
    create_distance_matrix,  # noqa: F401
    solve_1_5_approx,  # noqa: F401
    solve_2_approx,  # noqa: F401
    solve_greedy,  # noqa: F401
)
from mage.geography.vehicle_routing import (
    InvalidDepotException,  # noqa: F401
    VRPPath,  # noqa: F401
    VRPResult,  # noqa: F401
    VRPSolver,  # noqa: F401
)
