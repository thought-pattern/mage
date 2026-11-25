"""Temporal property codecs shared by export_util and import_util, so an export and its re-import use one encoding."""

from datetime import date, datetime, time, timedelta

from mage.export_import_util.duration import to_duration_iso_format
from mage.export_import_util.parameters import Parameter


def convert_to_isoformat(value: object):
    """
    Encodes a temporal property as the typed wrapper text the JSON export writes and the JSON import decodes, such as
    localDateTime(2024-01-02T03:04:05); any other value is returned unchanged.
    """
    if isinstance(value, timedelta):
        computed_return_value = Parameter.DURATION.value + str(value) + ")"
        return computed_return_value

    elif isinstance(value, time):
        computed_return_value = Parameter.LOCALTIME.value + value.isoformat() + ")"
        return computed_return_value

    elif isinstance(value, datetime):
        computed_return_value = Parameter.LOCALDATETIME.value + value.isoformat() + ")"
        return computed_return_value

    elif isinstance(value, date):
        computed_return_value = Parameter.DATE.value + value.isoformat() + ")"
        return computed_return_value

    else:
        return value


def convert_to_isoformat_graphML(value: object):
    """Encodes a temporal property as GraphML text (ISO-8601, durations as P...); any other value is returned unchanged."""
    if isinstance(value, timedelta):
        computed_return_value = to_duration_iso_format(value)
        return computed_return_value

    if isinstance(value, (time, date, datetime)):
        computed_return_value = value.isoformat()
        return computed_return_value

    else:
        return value
