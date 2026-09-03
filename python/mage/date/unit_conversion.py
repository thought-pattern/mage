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


def to_int(duration: datetime_timedelta, unit: str) -> int:
    if unit in Units.MILLISECOND:
        computed_return_value = int(duration / datetime_timedelta(milliseconds=1))
        return computed_return_value
    elif unit in Units.SECOND:
        computed_return_value = int(duration.total_seconds())
        return computed_return_value
    elif unit in Units.MINUTE:
        computed_return_value = int(duration / datetime_timedelta(minutes=1))
        return computed_return_value
    elif unit in Units.HOUR:
        computed_return_value = int(duration / datetime_timedelta(hours=1))
        return computed_return_value
    elif unit in Units.DAY:
        computed_return_value = int(duration / datetime_timedelta(days=1))
        return computed_return_value
    else:
        raise TypeError(f"The unit {unit} is not correct.")
    return 0
