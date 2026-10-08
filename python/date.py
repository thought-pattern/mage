"""Utilities for date."""

from datetime import datetime as datetime_datetime, timedelta as datetime_timedelta, timezone as datetime_timezone
from enum import IntEnum
from re import sub as re_sub
from zoneinfo import ZoneInfo

from mgp import (
    List as mgp_List,
    Nullable as mgp_Nullable,
    ProcCtx as mgp_ProcCtx,
    Record as mgp_Record,
    function as mgp_function,
    read_proc as mgp_read_proc,
)
from pytz import all_timezones as pytz_all_timezones, timezone as pytz_timezone

from mage.date.constants import Epoch
from mage.date.unit_conversion import to_int, to_timedelta


@mgp_read_proc
def parse(
    time: str,
    unit: str = "ms",
    format: str = "%Y-%m-%d %H:%M:%S",
    timezone: str = "UTC",
) -> mgp_Record(parsed=int):
    first_date = Epoch.UNIX_EPOCH.replace(tzinfo=datetime_timezone.utc)
    input_date = datetime_datetime.strptime(time, format)

    if timezone not in pytz_all_timezones:
        raise Exception("Timezone doesn't exist. Check documentation to see available timezones.")

    # A format carrying %z fixes its own instant. A naive wall time is placed in the named zone; a repeated (fall-back)
    # or skipped (spring-forward) wall time resolves to the zone's standard-time offset.
    if input_date.tzinfo is None:
        input_date = pytz_timezone(timezone).localize(input_date, is_dst=False)

    # Floor division over the complete duration keeps sub-unit precision (including microseconds) until the final
    # rounding, and floors instants before the epoch toward the past for every unit.
    time_since = input_date - first_date

    if unit == "ms":
        parsed = time_since // datetime_timedelta(milliseconds=1)
    elif unit == "s":
        parsed = time_since // datetime_timedelta(seconds=1)
    elif unit == "m":
        parsed = time_since // datetime_timedelta(minutes=1)
    elif unit == "h":
        parsed = time_since // datetime_timedelta(hours=1)
    elif unit == "d":
        parsed = time_since // datetime_timedelta(days=1)
    else:
        raise Exception("Unit doesn't exist. Check documentation to see available units.")

    computed_return_value = mgp_Record(parsed=parsed)
    return computed_return_value


@mgp_read_proc
def format(
    time: int,
    unit: str = "ms",
    format: str = "%Y-%m-%d %H:%M:%S %Z",
    timezone: str = "UTC",
) -> mgp_Record(formatted=str):
    first_date = Epoch.UNIX_EPOCH.replace(tzinfo=datetime_timezone.utc)

    if unit == "ms":
        new_date = first_date + datetime_timedelta(milliseconds=time)
    elif unit == "s":
        new_date = first_date + datetime_timedelta(seconds=time)
    elif unit == "m":
        new_date = first_date + datetime_timedelta(minutes=time)
    elif unit == "h":
        new_date = first_date + datetime_timedelta(hours=time)
    elif unit == "d":
        new_date = first_date + datetime_timedelta(days=time)
    else:
        raise Exception("Unit doesn't exist. Check documentation to see available units.")

    if timezone not in pytz_all_timezones:
        raise Exception("Timezone doesn't exist. Check documentation to see available timezones.")
    # The epoch offset is a UTC instant, so one UTC-to-zone conversion selects the offset in force at that instant.
    local_date = new_date.astimezone(pytz_timezone(timezone))

    computed_return_value = mgp_Record(formatted=local_date.strftime(format))
    return computed_return_value


@mgp_function
def add(
    time: int,
    unit: str,
    add_value: int,
    add_unit: str,
) -> int:
    computed_return_value = to_int(
        to_timedelta(time=time, unit=unit) + to_timedelta(time=add_value, unit=add_unit),
        unit=unit,
    )
    return computed_return_value


# TODO(colinbarry) Code below is a copy and paste from `date.py` in the Memgraph
# repo. This is a temporary fix to make it possible to use `date.convert_format`
# from the Memgraph+MAGE image; otherwise, MAGE's `date.py` will overwrite the
# Memgraph one, and users would lose the function. As the function is needed
# by GraphQL, and it seems some users are wanting to use GraphQL + MAGE, this
# seems the best approach for now. At some point, we will be moving to a
# monorepo, and the code below and tests in `date_test/test_convert_format` can
# be removed in favour of the copy in the Memgraph repo.


class FormatLength(IntEnum):
    """Enum for various date/time format lengths to replace magic numbers"""

    DATE = 10  # Length of 'YYYY-MM-DD'
    TIME = 8  # Length of 'HH:MM:SS'
    OFFSET = 5  # Length of '+hhmm' or '-hhmm'


class DateFormatUtil:
    """
    Utility class for converting between predefined ISO date formats using Python strftime and strptime.
    """

    ISO_DATE_FORMATS: dict[str, str] = {
        "basic_iso_date": "%Y%m%d",  # BASIC_ISO_DATE: '20111203'
        "iso_local_date": "%Y-%m-%d",  # ISO_LOCAL_DATE: '2011-12-03'
        "iso_offset_date": "%Y-%m-%d%z",  # ISO_OFFSET_DATE: '2011-12-03+01:00'
        "iso_date": "%Y-%m-%d",  # ISO_DATE: '2011-12-03' or '2011-12-03+01:00' (handled separately)
        "iso_local_time": "%H:%M:%S",  # ISO_LOCAL_TIME: '10:15:30'
        "iso_offset_time": "%H:%M:%S%z",  # ISO_OFFSET_TIME: '10:15:30+01:00'
        "iso_time": "%H:%M:%S",  # ISO_TIME: '10:15:30' or '10:15:30+01:00' (handled separately)
        "iso_local_date_time": "%Y-%m-%dT%H:%M:%S",  # ISO_LOCAL_DATE_TIME: '2011-12-03T10:15:30'
        "iso_offset_date_time": "%Y-%m-%dT%H:%M:%S%z",  # ISO_OFFSET_DATE_TIME: '2011-12-03T10:15:30+01:00'
        "iso_zoned_date_time": "iso_zoned_date_time",  # Special case
        "iso_date_time": "%Y-%m-%dT%H:%M:%S",  # ISO_DATE_TIME: '2011-12-03T10:15:30+01:00[Europe/Paris]' handled as zoned
    }

    @staticmethod
    def get_format(format_str: str) -> str:
        format_lower = format_str.lower()
        if format_lower == "iso_zoned_date_time" or format_lower == "iso_date_time":
            return "iso_zoned_date_time"
        if format_lower not in DateFormatUtil.ISO_DATE_FORMATS:
            raise ValueError(f"Unsupported date format: {format_str}")
        computed_return_value = DateFormatUtil.ISO_DATE_FORMATS.get(format_lower, "")
        return computed_return_value


@mgp_function
def convert_format(temporal: mgp_Nullable[str], current_format: str, convert_to: str) -> mgp_Nullable[str]:
    """
    Converts between specified ISO date formats using Python strftime and strptime.
    Supports zoned to offset conversion by removing zone part in '[]'.
    Offset to zoned returns the same string.
    Throws if parsing fails.

    Args:
        temporal: The datetime string to convert
        current_format: The current format of the datetime string
        convert_to: The target format to convert to

    Returns:
        output: The converted datetime string, or null if input is absent or empty
    """
    # Absent or blank input maps to Python None, which the host returns as Cypher null for this Nullable[str] function.
    if temporal is None or temporal.strip() == "":
        return None

    try:
        current_formatter = DateFormatUtil.get_format(current_format)
        convert_to_formatter = DateFormatUtil.get_format(convert_to)

        # Parse input string
        if current_formatter == "iso_zoned_date_time":
            # Remove zone part in [] and parse
            temporal_without_zone = temporal.split("[")[0]

            if "." in temporal_without_zone:
                temporal_without_zone = re_sub(r"(\.\d{6})\d*", r"\1", temporal_without_zone)
                dt = datetime_datetime.strptime(temporal_without_zone, "%Y-%m-%dT%H:%M:%S.%f%z")
            else:
                dt = datetime_datetime.strptime(temporal_without_zone, "%Y-%m-%dT%H:%M:%S%z")
        elif current_format.lower() == "iso_date":
            # iso_date can have optional offset, try parsing with offset first
            try:
                dt = datetime_datetime.strptime(temporal, "%Y-%m-%d%z")
            except ValueError:
                dt = datetime_datetime.strptime(temporal, "%Y-%m-%d")
        elif current_format.lower() == "iso_time":
            # iso_time can have optional offset
            try:
                dt = datetime_datetime.strptime(temporal, "%H:%M:%S%z")
            except ValueError:
                dt = datetime_datetime.strptime(temporal, "%H:%M:%S")
        else:
            try:
                dt = datetime_datetime.fromisoformat(temporal)
            except Exception:
                dt = datetime_datetime.strptime(temporal, current_formatter)

        if convert_to.lower() in ["iso_offset_date", "iso_offset_time", "iso_offset_date_time"] and dt.tzinfo is None:
            raise Exception("missing timezone")

        # Convert to target format
        if convert_to_formatter == "iso_zoned_date_time":
            # Converting to zoned date time: return offset datetime string (no zone name)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            converted = dt.isoformat()

        elif convert_to.lower() == "iso_date":
            # iso_date: include offset if timezone info is present
            if dt.tzinfo is not None:
                converted = dt.strftime("%Y-%m-%d%z")
                # Format offset as +hh:mm
                if len(converted) > FormatLength.DATE:
                    converted = f"{converted[:-2]}:{converted[-2:]}"
            else:
                converted = dt.strftime("%Y-%m-%d")
        elif convert_to.lower() == "iso_time":
            # iso_time: include offset if timezone info is present
            if dt.tzinfo is not None:
                converted = dt.strftime("%H:%M:%S%z")
                # Format offset as +hh:mm
                if len(converted) > FormatLength.TIME:
                    converted = f"{converted[:-2]}:{converted[-2:]}"
            else:
                converted = dt.strftime("%H:%M:%S")
        elif convert_to.lower() in [
            "iso_zoned_date_time",
            "iso_offset_date_time",
        ]:
            converted = dt.isoformat()
        else:
            # For offset formats, ensure timezone is present
            if convert_to_formatter.endswith("%z") and dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            # For local formats, remove timezone info
            elif not convert_to_formatter.endswith("%z") and dt.tzinfo is not None:
                dt = datetime_datetime(
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                    dt.microsecond,
                    fold=dt.fold,
                )

            converted = dt.strftime(convert_to_formatter)

            # Format offset as +hh:mm for offset formats
            if convert_to_formatter.endswith("%z") and len(converted) > FormatLength.DATE:
                converted = f"{converted[:-2]}:{converted[-2:]}"

        return converted

    except Exception as err:
        raise Exception(f"Error converting '{temporal}' from '{current_format}' to '{convert_to}': {err}") from err


@mgp_read_proc
def get_date_formats(context: mgp_ProcCtx) -> mgp_Record(formats=mgp_List[str]):
    """
    Returns a list of supported date formats.

    Returns:
        formats: List of supported date formats
    """
    computed_return_value = mgp_Record(formats=list(DateFormatUtil.ISO_DATE_FORMATS.keys()))
    return computed_return_value
