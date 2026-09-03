"""Utilities for constants."""

from datetime import datetime as datetime_datetime


class Conversion(int):
    MINUTES_IN_HOUR = 60
    SECONDS_IN_MINUTE = 60
    MILLISECONDS_IN_SECOND = 1000
    HOURS_IN_DAY = 24


class Epoch(datetime_datetime):
    UNIX_EPOCH = datetime_datetime(1970, 1, 1, 0, 0, 0)


class Units:
    MILLISECOND: set[str] = {"ms", "milli", "millis", "milliseconds"}
    SECOND: set[str] = {"s", "second", "seconds"}
    MINUTE: set[str] = {"m", "minute", "minutes"}
    HOUR: set[str] = {"h", "hour", "hours"}
    DAY: set[str] = {"d", "day", "days"}
