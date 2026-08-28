"""Utilities for constants."""

from datetime import datetime as datetime_datetime
from typing import ClassVar as typing_ClassVar


class Conversion(int):
    MINUTES_IN_HOUR = 60
    SECONDS_IN_MINUTE = 60
    MILLISECONDS_IN_SECOND = 1000
    HOURS_IN_DAY = 24


class Epoch(datetime_datetime):
    UNIX_EPOCH = datetime_datetime(1970, 1, 1, 0, 0, 0)


class Units:
    MILLISECOND: typing_ClassVar = {"ms", "milli", "millis", "milliseconds"}
    SECOND: typing_ClassVar = {"s", "second", "seconds"}
    MINUTE: typing_ClassVar = {"m", "minute", "minutes"}
    HOUR: typing_ClassVar = {"h", "hour", "hours"}
    DAY: typing_ClassVar = {"d", "day", "days"}
