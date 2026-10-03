"""Duration codecs shared by export_util and import_util."""

from datetime import timedelta

ZERO_DURATION = timedelta(0)


def duration_components(value: timedelta) -> tuple[int, int, int, int, int]:
    """
    Splits a timedelta into signed (days, hours, minutes, seconds, microseconds). The components come from the magnitude
    of the whole value and all carry its sign, so -1 second is (0, 0, 0, -1, 0) rather than Python's normalized
    (-1 day, +86399 seconds).
    """
    sign = -1 if value < ZERO_DURATION else 1
    magnitude = abs(value)
    minutes, seconds = divmod(magnitude.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    components = (
        sign * magnitude.days,
        sign * hours,
        sign * minutes,
        sign * seconds,
        sign * magnitude.microseconds,
    )
    return components


def to_duration_iso_format(value: timedelta) -> str:
    """
    Encodes a timedelta as ISO-8601 style text P[nD][T[nH][nM][n[.ffffff]S]], as Memgraph's duration() writes it.

    Memgraph rejects a sign before P, so a negative value carries its sign on every non-zero component, the form
    Memgraph's own Duration::ToString writes (one second less than zero is PT-1S). Fractional seconds always have six
    digits (microseconds), and zero is PT0S.
    """
    days, hours, minutes, seconds, microseconds = duration_components(value)
    sign = "-" if value < ZERO_DURATION else ""

    date_part = f"{days}D" if days else ""
    time_parts = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if seconds or microseconds:
        fraction = f".{abs(microseconds):06d}" if microseconds else ""
        time_parts.append(f"{sign}{abs(seconds)}{fraction}S")
    if not date_part and not time_parts:
        time_parts.append("0S")

    time_part = f"T{''.join(time_parts)}" if time_parts else ""
    encoded = f"P{date_part}{time_part}"
    return encoded


def to_cypher_duration(value: timedelta) -> str:
    """
    Encodes a timedelta as a Memgraph duration() expression that reconstructs it exactly. The map form is used because
    Memgraph parses the seconds of a duration string as a double and truncates it, losing a microsecond for about one
    fractional value in a hundred; integer map components are exact.
    """
    days, hours, minutes, seconds, microseconds = duration_components(value)
    expression = f"duration({{day: {days}, hour: {hours}, minute: {minutes}, second: {seconds}, microsecond: {microseconds}}})"
    return expression
