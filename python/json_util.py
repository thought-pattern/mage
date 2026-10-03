"""Utilities for json util."""

from datetime import date, datetime, time, timedelta
from io import DEFAULT_BUFFER_SIZE, TextIOWrapper
from json import dumps as json_dumps
from json import load as json_load
from json import loads as json_loads
from pathlib import Path
from time import monotonic as time_monotonic
from urllib.request import Request, urlopen

from mgp import Edge as mgp_Edge
from mgp import Nullable as mgp_Nullable
from mgp import Path as mgp_Path
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import function as mgp_function
from mgp import read_proc as mgp_read_proc

# A URL source is remote and unsized until it is read. These bounds own its acquisition: the whole fetch must finish
# within the deadline (each blocking socket operation also times out at it), and the body may not exceed the byte
# allowance. JSON parsing is linear in the admitted body, so the allowance bounds it too.
JSON_URL_TIMEOUT_SECONDS = 60.0
JSON_URL_MAX_BYTES = 64 * 1024 * 1024


def convert_value_to_json_compatible(value: object) -> object:
    """Helper function to convert Memgraph values to JSON-compatible Python types."""
    if isinstance(value, mgp_Vertex):
        computed_return_value = {
            "type": "node",
            "id": value.id,
            "labels": [label.name for label in value.labels],
            "properties": {k: convert_value_to_json_compatible(v) for k, v in value.properties.items()},
        }
        return computed_return_value
    elif isinstance(value, mgp_Edge):
        computed_return_value = {
            "type": "relationship",
            "id": value.id,
            "start": value.from_vertex.id,
            "end": value.to_vertex.id,
            "relationship_type": value.type.name,
            "properties": {k: convert_value_to_json_compatible(v) for k, v in value.properties.items()},
        }
        return computed_return_value
    elif isinstance(value, mgp_Path):
        computed_return_value = {
            "type": "path",
            "start": convert_value_to_json_compatible(value.vertices[0]),
            "end": convert_value_to_json_compatible(value.vertices[-1]),
            "nodes": [convert_value_to_json_compatible(v) for v in value.vertices],
            "relationships": [convert_value_to_json_compatible(e) for e in value.edges],
        }
        return computed_return_value
    elif isinstance(value, (list, tuple)):
        computed_return_value = [convert_value_to_json_compatible(item) for item in value]
        return computed_return_value
    elif isinstance(value, dict):
        computed_return_value = {k: convert_value_to_json_compatible(v) for k, v in value.items()}
        return computed_return_value
    elif isinstance(value, (datetime, date)) or isinstance(value, time):
        computed_return_value = value.isoformat()
        return computed_return_value
    elif isinstance(value, timedelta):
        total_seconds = value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        microseconds = value.microseconds
        computed_return_value = f"P0DT{hours}H{minutes}M{seconds}.{microseconds:06d}S"
        return computed_return_value
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        computed_return_value = str(value)
        return computed_return_value


def extract_objects(file: TextIOWrapper):
    """Helper function to extract objects from a JSON file."""
    objects = json_load(file)
    if type(objects) is dict:
        objects = [objects]
    return objects


@mgp_function
def to_json(value: object):
    converted = convert_value_to_json_compatible(value)
    computed_return_value = json_dumps(converted, ensure_ascii=False)
    return computed_return_value


@mgp_function
def from_json_list(json_str: mgp_Nullable[str]):
    if json_str is None:
        # External null stays Cypher null; the host maps Python None back to null.
        return json_str

    value = json_loads(json_str)
    if not isinstance(value, list):
        raise ValueError("Input JSON must represent a list")
    return value


@mgp_read_proc
def load_from_path(ctx: mgp_ProcCtx, path: str) -> mgp_Record:
    file = Path(path)
    if file.exists():
        with file.open() as opened_file:
            objects = extract_objects(opened_file)
    else:
        raise FileNotFoundError("There is no file " + path)

    computed_return_value = mgp_Record(objects=objects)
    return computed_return_value


@mgp_read_proc
def load_from_str(ctx: mgp_ProcCtx, json_str: str) -> mgp_Record:
    """
    Procedure to load JSON from a string.

    Parameters
    ----------
    json_str : str
        JSON string that is being loaded.
    """
    objects = json_loads(json_str)
    if type(objects) is dict:
        objects = [objects]

    computed_return_value = mgp_Record(objects=objects)
    return computed_return_value


def fetch_json_bytes(ctx: mgp_ProcCtx, url: str) -> bytes:
    """
    Fetches a JSON document within JSON_URL_TIMEOUT_SECONDS and JSON_URL_MAX_BYTES, closing the response on every
    path. read1 performs at most one blocking read per call, so the deadline and host abort are checked between reads.
    """
    request = Request(url, headers={"User-Agent": "MAGE module"})
    deadline = time_monotonic() + JSON_URL_TIMEOUT_SECONDS
    chunks: list[bytes] = []
    received_bytes = 0
    try:
        with urlopen(request, timeout=JSON_URL_TIMEOUT_SECONDS) as response:
            while True:
                ctx.check_must_abort()
                chunk = response.read1(DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                received_bytes += len(chunk)
                if received_bytes > JSON_URL_MAX_BYTES:
                    raise ValueError(f"JSON response from {url} exceeds {JSON_URL_MAX_BYTES} bytes")
                if time_monotonic() > deadline:
                    raise TimeoutError(f"JSON response from {url} did not complete within {JSON_URL_TIMEOUT_SECONDS} seconds")
                chunks.append(chunk)
    except OSError as err:
        raise ValueError(f"Error while fetching JSON from {url}: {err}") from err
    json_bytes = b"".join(chunks)
    return json_bytes


@mgp_read_proc
def load_from_url(ctx: mgp_ProcCtx, url: str) -> mgp_Record:
    objects = json_loads(fetch_json_bytes(ctx, url))
    if type(objects) is dict:
        objects = [objects]

    computed_return_value = mgp_Record(objects=objects)
    return computed_return_value
