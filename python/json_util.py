"""Utilities for json util."""

from datetime import date, datetime, time, timedelta
from io import TextIOWrapper
from json import dumps as json_dumps
from json import load as json_load
from json import loads as json_loads
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from mgp import Edge as mgp_Edge
from mgp import List as mgp_List
from mgp import Nullable as mgp_Nullable
from mgp import Path as mgp_Path
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import function as mgp_function
from mgp import read_proc as mgp_read_proc


def _convert_value_to_json_compatible(value: Any) -> Any:
    """Helper function to convert Memgraph values to JSON-compatible Python types."""
    if isinstance(value, mgp_Vertex):
        _return_value = {
            "type": "node",
            "id": value.id,
            "labels": [label.name for label in value.labels],
            "properties": {
                k: _convert_value_to_json_compatible(v)
                for k, v in value.properties.items()
            },
        }
        return _return_value
    elif isinstance(value, mgp_Edge):
        _return_value = {
            "type": "relationship",
            "id": value.id,
            "start": value.from_vertex.id,
            "end": value.to_vertex.id,
            "relationship_type": value.type.name,
            "properties": {
                k: _convert_value_to_json_compatible(v)
                for k, v in value.properties.items()
            },
        }
        return _return_value
    elif isinstance(value, mgp_Path):
        _return_value = {
            "type": "path",
            "start": _convert_value_to_json_compatible(value.vertices[0]),
            "end": _convert_value_to_json_compatible(value.vertices[-1]),
            "nodes": [_convert_value_to_json_compatible(v) for v in value.vertices],
            "relationships": [
                _convert_value_to_json_compatible(e) for e in value.edges
            ],
        }
        return _return_value
    elif isinstance(value, (list, tuple)):
        _return_value = [_convert_value_to_json_compatible(item) for item in value]
        return _return_value
    elif isinstance(value, dict):
        _return_value = {
            k: _convert_value_to_json_compatible(v) for k, v in value.items()
        }
        return _return_value
    elif isinstance(value, (datetime, date)) or isinstance(value, time):
        _return_value = value.isoformat()
        return _return_value
    elif isinstance(value, timedelta):
        total_seconds = value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        microseconds = value.microseconds
        _return_value = f"P0DT{hours}H{minutes}M{seconds}.{microseconds:06d}S"
        return _return_value
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        _return_value = str(value)
        return _return_value


def extract_objects(file: TextIOWrapper):
    """Helper function to extract objects from a JSON file."""
    objects = json_load(file)
    if type(objects) is dict:
        objects = [objects]
    return objects


@mgp_function
def to_json(value: Any):
    converted = _convert_value_to_json_compatible(value)
    _return_value = json_dumps(converted, ensure_ascii=False)
    return _return_value


@mgp_function
def from_json_list(json_str: mgp_Nullable[str]):
    if json_str is None:
        return False

    value = json_loads(json_str)
    if not isinstance(value, list):
        raise ValueError("Input JSON must represent a list")
    return value


@mgp_read_proc
def load_from_path(ctx: mgp_ProcCtx, path: str) -> mgp_Record(objects=mgp_List[object]):
    file = Path(path)
    if file.exists():
        with file.open() as opened_file:
            objects = extract_objects(opened_file)
    else:
        raise FileNotFoundError("There is no file " + path)

    _return_value = mgp_Record(objects=objects)
    return _return_value


@mgp_read_proc
def load_from_str(ctx: mgp_ProcCtx, json_str: str) -> mgp_Record(
    objects=mgp_List[object]
):
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

    _return_value = mgp_Record(objects=objects)
    return _return_value


@mgp_read_proc
def load_from_url(ctx: mgp_ProcCtx, url: str) -> mgp_Record(objects=mgp_List[object]):
    request = Request(url)
    request.add_header("User-Agent", "MAGE module")
    try:
        content = urlopen(request)
    except URLError as url_error:
        raise url_error from url_error
    else:
        objects = extract_objects(content)

    _return_value = mgp_Record(objects=objects)
    return _return_value
