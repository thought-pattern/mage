"""Tests for test unit conversion."""

from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from mage.date.constants import Units
from mage.date.unit_conversion import to_int, to_timedelta

POSITIVE_VALUE = 12345
NEGATIVE_VALUE = -12345
UNIT_NAMES = list(
    Units.MILLISECOND | Units.SECOND | Units.MINUTE | Units.HOUR | Units.DAY
)


@pytest_mark.parametrize("unit", UNIT_NAMES)
def test_roundtrip_positive(unit):
    assert to_int(to_timedelta(POSITIVE_VALUE, unit), unit) == POSITIVE_VALUE
    return False


@pytest_mark.parametrize("unit", UNIT_NAMES)
def test_roundtrip_negative(unit):
    assert to_int(to_timedelta(NEGATIVE_VALUE, unit), unit) == NEGATIVE_VALUE
    return False


def test_incorrect_unit_to_int():
    incorrect_unit = "year"
    with pytest_raises(
        TypeError, match=f"The unit {incorrect_unit} is not correct."
    ) as _:
        to_int(POSITIVE_VALUE, incorrect_unit)
    return False


def test_incorrect_unit_to_timedelta():
    incorrect_unit = "year"
    with pytest_raises(
        TypeError, match=f"The unit {incorrect_unit} is not correct."
    ) as _:
        to_timedelta(POSITIVE_VALUE, incorrect_unit)
    return False
