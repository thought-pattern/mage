"""Utilities for temporal."""

from datetime import date as datetime_date
from datetime import datetime as datetime_datetime
from datetime import time as datetime_time
from datetime import timedelta as datetime_timedelta

from mgp import Any as mgp_Any
from mgp import Record as mgp_Record
from mgp import read_proc as mgp_read_proc

from mage.date.constants import Epoch


@mgp_read_proc
def format(
    temporal: mgp_Any,
    format: str = "ISO",
) -> mgp_Record:
    if not (
        isinstance(temporal, datetime_datetime)
        or isinstance(temporal, datetime_date)
        or isinstance(temporal, datetime_time)
        or isinstance(temporal, datetime_timedelta)
    ):
        computed_return_value = mgp_Record(formatted=str(temporal))
        return computed_return_value

    if "%z" in format or "%Z" in format:
        raise Exception(
            "Memgraph works with UTC zone only so\
            '%Z' in format is not supported."
        )

    if format == "ISO" and (
        isinstance(temporal, datetime_datetime) or isinstance(temporal, datetime_date) or isinstance(temporal, datetime_time)
    ):
        computed_return_value = mgp_Record(formatted=temporal.isoformat())
        return computed_return_value

    if isinstance(temporal, datetime_timedelta):
        temporal = Epoch.UNIX_EPOCH + temporal

    computed_return_value = mgp_Record(formatted=temporal.strftime(format))
    return computed_return_value
