"""Temporal property codecs shared by export_util and import_util, so an export and its re-import use one encoding."""

from datetime import date, datetime, time, timedelta

from mage.export_import_util.duration import to_duration_iso_format
from mage.export_import_util.parameters import Parameter


def convert_to_isoformat(property: object):
    """
    Encodes a temporal property as the typed wrapper text the JSON export writes and the JSON import decodes, such as
    localDateTime(2024-01-02T03:04:05); any other value is returned unchanged.
    """
    if isinstance(property, timedelta):
        computed_return_value = Parameter.DURATION.value + str(property) + ")"
        return computed_return_value

    elif isinstance(property, time):
        computed_return_value = Parameter.LOCALTIME.value + property.isoformat() + ")"
        return computed_return_value

    elif isinstance(property, datetime):
        computed_return_value = Parameter.LOCALDATETIME.value + property.isoformat() + ")"
        return computed_return_value

    elif isinstance(property, date):
        computed_return_value = Parameter.DATE.value + property.isoformat() + ")"
        return computed_return_value

    else:
        return property


def convert_to_isoformat_graphML(property: object):
    """Encodes a temporal property as GraphML text (ISO-8601, durations as P...); any other value is returned unchanged."""
    if isinstance(property, timedelta):
        computed_return_value = to_duration_iso_format(property)
        return computed_return_value

    if isinstance(property, (time, date, datetime)):
        computed_return_value = property.isoformat()
        return computed_return_value

    else:
        return property
