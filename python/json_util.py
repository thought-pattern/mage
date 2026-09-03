"""Utilities for json util."""

from datetime import date, datetime, time, timedelta
from io import TextIOWrapper
from json import dumps as json_dumps
from json import load as json_load
from json import loads as json_loads
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from mgp import Edge as mgp_Edge
from mgp import Nullable as mgp_Nullable
from mgp import Path as mgp_Path
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import function as mgp_function
from mgp import read_proc as mgp_read_proc


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
        return False

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


@mgp_read_proc
def load_from_url(ctx: mgp_ProcCtx, url: str) -> mgp_Record:
    request = Request(url)
    request.add_header("User-Agent", "MAGE module")
    try:
        content = urlopen(request)
    except URLError as url_error:
        raise url_error from url_error
    else:
        objects = extract_objects(content)

    computed_return_value = mgp_Record(objects=objects)
    return computed_return_value
