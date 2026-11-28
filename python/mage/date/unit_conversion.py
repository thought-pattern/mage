"""Utilities for unit conversion."""

from datetime import timedelta as datetime_timedelta

from mage.date.constants import Units


def to_timedelta(time: int, unit: str) -> datetime_timedelta:
    if unit in Units.MILLISECOND:
        computed_return_value = datetime_timedelta(milliseconds=time)
        return computed_return_value
    elif unit in Units.SECOND:
        computed_return_value = datetime_timedelta(seconds=time)
        return computed_return_value
    elif unit in Units.MINUTE:
        computed_return_value = datetime_timedelta(minutes=time)
        return computed_return_value
    elif unit in Units.HOUR:
        computed_return_value = datetime_timedelta(hours=time)
        return computed_return_value
    elif unit in Units.DAY:
        computed_return_value = datetime_timedelta(days=time)
        return computed_return_value
    else:
        raise TypeError(f"The unit {unit} is not correct.")


MICROSECONDS_PER_MILLISECOND = 1_000
MICROSECONDS_PER_SECOND = 1_000_000
MICROSECONDS_PER_MINUTE = 60_000_000
MICROSECONDS_PER_HOUR = 3_600_000_000
MICROSECONDS_PER_DAY = 86_400_000_000
SECONDS_PER_DAY = 86_400


def truncate_quotient(dividend: int, divisor: int) -> int:
    """Divides two integers with truncation toward zero and no floating-point intermediate."""
    magnitude = abs(dividend) // divisor
    quotient = -magnitude if dividend < 0 else magnitude
    return quotient


def to_int(duration: datetime_timedelta, unit: str) -> int:
    total_microseconds = (duration.days * SECONDS_PER_DAY + duration.seconds) * MICROSECONDS_PER_SECOND + duration.microseconds
    if unit in Units.MILLISECOND:
        computed_return_value = truncate_quotient(total_microseconds, MICROSECONDS_PER_MILLISECOND)
        return computed_return_value
    if unit in Units.SECOND:
        computed_return_value = truncate_quotient(total_microseconds, MICROSECONDS_PER_SECOND)
        return computed_return_value
    if unit in Units.MINUTE:
        computed_return_value = truncate_quotient(total_microseconds, MICROSECONDS_PER_MINUTE)
        return computed_return_value
    if unit in Units.HOUR:
        computed_return_value = truncate_quotient(total_microseconds, MICROSECONDS_PER_HOUR)
        return computed_return_value
    if unit in Units.DAY:
        computed_return_value = truncate_quotient(total_microseconds, MICROSECONDS_PER_DAY)
        return computed_return_value
    raise TypeError(f"The unit {unit} is not correct.")
