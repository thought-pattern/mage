"""Utilities for export util."""

from csv import QUOTE_ALL as csv_QUOTE_ALL
from csv import QUOTE_MINIMAL as csv_QUOTE_MINIMAL
from csv import QUOTE_NONE as csv_QUOTE_NONE
from csv import QUOTE_NONNUMERIC as csv_QUOTE_NONNUMERIC
from csv import Error as csv_Error
from csv import writer as csv_writer
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import StringIO as io_StringIO
from json import dump as js_dump
from json import dumps as js_dumps
from math import floor
from os import path as os_path
from typing import Any, Dict, List

from gqlalchemy import Memgraph
from gqlalchemy import Memgraph as gqlalchemy_Memgraph
from mgp import Any as mgp_Any
from mgp import Edge as mgp_Edge
from mgp import List as mgp_List
from mgp import Map as mgp_Map
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc

from mage.export_import_util.parameters import Parameter

_DEFAULT_ARGUMENT_DICT = {}
_DEFAULT_ARGUMENT_DICT_2 = {
    "graphML": False,
    "leaveOutLabels": False,
    "leaveOutProperties": False,
}

HEADER_FILENAME = "header.csv"


@dataclass
class Node:
    id: int
    labels: list
    properties: dict

    def get_dict(self) -> dict:
        return {
            Parameter.ID.value: self.id,
            Parameter.LABELS.value: self.labels,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.TYPE.value: Parameter.NODE.value,
        }


@dataclass
class Relationship:
    end: int
    id: int
    label: str
    properties: dict
    start: int
    id: int

    def get_dict(self) -> dict:
        return {
            Parameter.END.value: self.end,
            Parameter.ID.value: self.id,
            Parameter.LABEL.value: self.label,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.START.value: self.start,
            Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
        }


@dataclass
class KeyObjectGraphML:
    name: str
    is_for: str
    type: str
    type_is_list: bool
    default_value: str
    id: str = ""

    def __init__(
        self,
        name: str,
        is_for: str,
        type: str = "",
        type_is_list: str = "",
        default_value: str = "",
    ):
        self.name = name
        self.is_for = is_for
        self.type = type
        self.type_is_list = type_is_list
        self.default_value = default_value

    def __hash__(self):
        _return_value = hash(
            (
                self.name,
                self.is_for,
                self.type,
                self.type_is_list,
                self.default_value,
            )
        )
        return _return_value

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        _return_value = (
            self.name == other.name
            and self.is_for == other.is_for
            and self.type == other.type
            and self.type_is_list == other.type_is_list
            and self.default_value == other.default_value
        )
        return _return_value


def convert_to_isoformat(property: object):
    if isinstance(property, timedelta):
        _return_value = Parameter.DURATION.value + str(property) + ")"
        return _return_value

    elif isinstance(property, time):
        _return_value = Parameter.LOCALTIME.value + property.isoformat() + ")"
        return _return_value

    elif isinstance(property, datetime):
        _return_value = Parameter.LOCALDATETIME.value + property.isoformat() + ")"
        return _return_value

    elif isinstance(property, date):
        _return_value = Parameter.DATE.value + property.isoformat() + ")"
        return _return_value

    else:
        return property


def to_duration_iso_format(value: timedelta) -> str:
    """Converts timedelta to ISO-8601 duration: P<date>T<time>"""
    date_parts: List[str] = []
    time_parts: List[str] = []

    if value.days != 0:
        date_parts.append(f"{abs(value.days)}D")

    if value.seconds != 0 or value.microseconds != 0:
        abs_seconds = abs(value.seconds)
        hours = floor(abs_seconds / 3600)
        minutes = floor((abs_seconds - hours * 3600) / 60)
        seconds = abs_seconds - hours * 3600 - minutes * 60
        microseconds = value.microseconds

        if hours > 0:
            time_parts.append(f"{hours}H")
        if minutes > 0:
            time_parts.append(f"{minutes}M")
        if seconds > 0 or microseconds > 0:
            microseconds_part = (
                f".{abs(value.microseconds)}" if value.microseconds != 0 else ""
            )
            time_parts.append(f"{seconds}{microseconds_part}S")

    date_duration_str = "".join(date_parts)
    time_duration_str = f"T{''.join(time_parts)}" if time_parts else ""

    _return_value = f"P{date_duration_str}{time_duration_str}"
    return _return_value


def convert_to_cypher_format(property: object) -> str:
    if isinstance(property, timedelta):
        _return_value = f"duration('{to_duration_iso_format(property)}')"
        return _return_value

    elif isinstance(property, time):
        _return_value = f"localTime('{property.isoformat()}')"
        return _return_value

    elif isinstance(property, datetime):
        _return_value = f"localDateTime('{property.isoformat()}')"
        return _return_value

    elif isinstance(property, date):
        _return_value = f"date('{property.isoformat()}')"
        return _return_value

    elif isinstance(property, str):
        _return_value = f"'{property}'"
        return _return_value

    elif isinstance(property, tuple):  # list
        _return_value = (
            "[" + ", ".join([convert_to_cypher_format(item) for item in property]) + "]"
        )
        return _return_value

    elif isinstance(property, dict):
        _return_value = (
            "{"
            + ", ".join(
                [f"{k}: {convert_to_cypher_format(v)}" for k, v in property.items()]
            )
            + "}"
        )
        return _return_value

    _return_value = str(property)
    return _return_value


def get_properties_cypher(object, write_properties: bool) -> dict:
    _return_value = (
        {
            key: convert_to_cypher_format(object.properties.get(key, False))
            for key in object.properties.keys()
        }
        if write_properties
        else {}
    )
    return _return_value


def get_graph_for_cypher(ctx: mgp_ProcCtx, write_properties: bool) -> List[object]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = get_properties_cypher(vertex, write_properties)
        nodes.append(Node(vertex.id, labels, properties))

        for edge in vertex.out_edges:
            properties = get_properties_cypher(edge, write_properties)
            relationships.append(
                Relationship(
                    edge.to_vertex.id,
                    edge.id,
                    edge.type.name,
                    properties,
                    edge.from_vertex.id,
                )
            )

    _return_value = nodes + relationships
    return _return_value


def format_properties_cypher(properties) -> str:
    _return_value = "{" + ", ".join([f"{k}: {v}" for k, v in properties.items()]) + "}"
    return _return_value


@mgp_read_proc
def cypher_all(
    ctx: mgp_ProcCtx,
    path: str = "",
    config: mgp_Map = _DEFAULT_ARGUMENT_DICT,
) -> mgp_Record(path=str, data=str):
    (
        "Exports the graph in cypher with all the constraints, indexes and triggers.\n    Args:\n      "  # Continue literal.
        "  context (mgp.ProcCtx): Reference to the context execution.\n        path (str): A path to t"  # Continue literal.
        "he file where the query results will be exported. Defaults to an empty string.\n        confi"  # Continue literal.
        "g : mgp.Map\n            stream (bool) = False: Flag to export the graph data to a stream.\n  "  # Continue literal.
        "          write_properties (bool) = True: Flag to keep node and relationship properties. By "  # Continue literal.
        "default set to true.\n            write_triggers (bool) = True: Flag to export graph triggers"  # Continue literal.
        ".\n            write_indexes (bool) = True: Flag to export indexes.\n            write_constra"  # Continue literal.
        "ints (bool) = True: Flag to export constraints.\n    Returns:\n        path (str): A path to t"  # Continue literal.
        "he file where the query results are exported. If path is not provided, the output will be an"  # Continue literal.
        " empty string.\n        data (str): A stream of query results in a cypher format.\n    Raises:"  # Continue literal.
        "\n        PermissionError: If you provided file path that you have no permissions to write at"  # Continue literal.
        ".\n        OSError: If the file can't be opened or written to.\n"
    )

    if config is _DEFAULT_ARGUMENT_DICT:
        config = _DEFAULT_ARGUMENT_DICT.copy()
    cypher = []

    memgraph = gqlalchemy_Memgraph()

    if config.get("write_triggers", True):
        triggers = memgraph.execute_and_fetch("SHOW TRIGGERS;")
        for trigger in triggers:
            trigger_name = trigger.get("trigger name", "")
            event_type = trigger.get("event type", "")
            phase = trigger.get("phase", "")
            statement = trigger.get("statement", "")
            cypher.append(
                f"CREATE TRIGGER {trigger_name} ON {event_type} {phase} EXECUTE {statement};"
            )
        cypher.append("")

    if config.get("write_indexes", True):
        constraints = memgraph.execute_and_fetch("SHOW CONSTRAINT INFO;")
        for constraint in constraints:
            constraint_type = constraint.get("constraint type", "")

            if constraint_type == "exists":
                cypher.append(
                    f"CREATE CONSTRAINT ON (n:{constraint.get('label', '')}) ASSERT EXISTS (n.{constraint.get('properties', [])});"
                )
            elif constraint_type == "unique":
                properties = (
                    [constraint.get("properties", [])]
                    if isinstance(constraint.get("properties", []), str)
                    else list(constraint.get("properties", []))
                )
                cypher.append(
                    f"CREATE CONSTRAINT ON (n:{constraint.get('label', '')}) ASSERT {'n.' + ', n.'.join(properties)} IS UNIQUE;"
                )
            else:
                raise ValueError("Unknown constraint type.")
        cypher.append("")

    if config.get("write_constraints", True):
        indexes = memgraph.execute_and_fetch("SHOW INDEX INFO;")
        for index in indexes:
            index_type = index.get("index type", "")
            if index_type == "label":
                cypher.append(f"CREATE INDEX ON :{index.get('label', '')};")
            elif index_type == "label+property":
                cypher.append(
                    f"CREATE INDEX ON :{index.get('label', '')}({index.get('property', False)});"
                )
            else:
                raise ValueError("Unknown index type.")
        cypher.append("")

    graph = get_graph_for_cypher(ctx, config.get("write_properties", True))

    for object in graph:
        if isinstance(object, Node):
            object.labels.append("_IMPORT_ID")
            object.properties["_IMPORT_ID"] = object.id
            properties_str = format_properties_cypher(object.properties)
            cypher.append(f"CREATE (n:{':'.join(object.labels)} {properties_str});")
        elif isinstance(object, Relationship):
            properties_str = format_properties_cypher(object.properties)
            cypher.append(
                f"MATCH (n:_IMPORT_ID {{_IMPORT_ID: {object.start}}}) MATCH (m:_IMPORT_ID {{_IMPORT_ID: "
                f"{object.end}}}) CREATE (n)-[:{object.label} {properties_str}]->(m);"
            )

    cypher.append("MATCH (n:_IMPORT_ID) REMOVE n:`_IMPORT_ID` REMOVE n._IMPORT_ID;")

    if path:
        try:
            with open(path, "w") as f:
                f.write("\n".join(cypher))
        except PermissionError as _caught_error_332:
            raise PermissionError(
                "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
            ) from _caught_error_332
        except Exception as _caught_error_336:
            raise OSError("Could not open or write to the file.") from _caught_error_336

    _return_value = mgp_Record(
        path=path, data="\n".join(cypher) if config.get("stream", False) else ""
    )
    return _return_value


def get_properties_json(object, write_properties: bool):
    _return_value = (
        {
            key: convert_to_isoformat(object.properties.get(key, False))
            for key in object.properties.keys()
        }
        if write_properties
        else {}
    )
    return _return_value


def convert_to_isoformat_graphML(property: object):
    if isinstance(property, timedelta):
        _return_value = to_duration_iso_format(property)
        return _return_value

    if isinstance(property, (time, date, datetime)):
        _return_value = property.isoformat()
        return _return_value

    else:
        return property


def get_graph(ctx: mgp_ProcCtx, write_properties: bool) -> List[object]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = get_properties_json(vertex, write_properties)

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            properties = get_properties_json(edge, write_properties)

            relationships.append(
                Relationship(
                    edge.to_vertex.id,
                    edge.id,
                    edge.type.name,
                    properties,
                    edge.from_vertex.id,
                ).get_dict()
            )

    _return_value = nodes + relationships
    return _return_value


def get_graphML(
    ctx: mgp_ProcCtx,
    config: mgp_Map = _DEFAULT_ARGUMENT_DICT_2,
) -> List[object]:
    """
    config : Map
        - graphML: bool
        - leaveOutLabels: bool
        - leaveOutProperties: bool

    """
    if config is _DEFAULT_ARGUMENT_DICT_2:
        config = _DEFAULT_ARGUMENT_DICT_2.copy()
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = []
        properties = dict()
        if not config.get("leaveOutLabels", []):
            labels = [label.name for label in vertex.labels]
        if config.get("graphML", False) and not config.get("leaveOutProperties", []):
            properties = {
                key: convert_to_isoformat_graphML(vertex.properties.get(key, False))
                for key in vertex.properties.keys()
            }
        elif not config.get("leaveOutProperties", []):
            properties = {
                key: convert_to_isoformat(vertex.properties.get(key, False))
                for key in vertex.properties.keys()
            }

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            if not config.get("leaveOutProperties", []):
                properties = {
                    key: convert_to_isoformat(edge.properties.get(key, False))
                    for key in edge.properties.keys()
                }

            relationships.append(
                Relationship(
                    edge.to_vertex.id,
                    edge.id,
                    edge.type.name,
                    properties,
                    edge.from_vertex.id,
                ).get_dict()
            )

    _return_value = nodes + relationships
    return _return_value


def get_graph_from_list(
    graph_vertices: list, graph_edges: list, write_properties: bool
) -> List[object]:
    nodes = list()
    relationships = list()

    for vertex in graph_vertices:
        labels = [label.name for label in vertex.labels]
        properties = get_properties_json(vertex, write_properties)

        nodes.append(Node(vertex.id, labels, properties).get_dict())

    for edge in graph_edges:
        properties = get_properties_json(edge, write_properties)

        relationships.append(
            Relationship(
                edge.to_vertex.id,
                edge.id,
                edge.type.name,
                properties,
                edge.from_vertex.id,
            ).get_dict()
        )

    _return_value = nodes + relationships
    return _return_value


def get_graph_info_from_lists(
    node_list: List[mgp_Vertex], relationship_list: List[mgp_Edge]
):
    graph = list()
    all_node_properties = list()
    all_node_prop_set = set()
    all_relationship_properties = list()
    all_relationship_prop_set = set()

    for node in node_list:
        for prop in node.properties:
            if prop not in all_node_prop_set:
                all_node_properties.append(prop)
                all_node_prop_set.add(prop)
        graph.append(Node(node.id, node.labels, node.properties))
    all_node_properties.sort()

    for relationship in relationship_list:
        for prop in relationship.properties:
            if prop not in all_relationship_prop_set:
                all_relationship_properties.append(prop)
                all_relationship_prop_set.add(prop)

        graph.append(
            Relationship(
                relationship.to_vertex.id,
                relationship.id,
                relationship.type.name,
                relationship.properties,
                relationship.from_vertex.id,
            )
        )
    all_relationship_properties.sort()

    return graph, all_node_properties, all_relationship_properties


def json_dump_to_file(graph: List[object], path: str):
    try:
        with open(path, "w") as outfile:
            js_dump(
                graph,
                outfile,
                indent=Parameter.STANDARD_INDENT.value,
                default=str,
            )
    except PermissionError as _caught_error_512:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        ) from _caught_error_512
    except Exception as _caught_error_517:
        raise OSError("Could not open or write to the file.") from _caught_error_517
    return False


@mgp_read_proc
def json(
    ctx: mgp_ProcCtx, path: str = "", config: mgp_Map = _DEFAULT_ARGUMENT_DICT
) -> mgp_Record(path=str, data=str):
    (
        "\n    Procedure to export the whole database to a JSON file.\n\n    Parameters:\n        context"  # Continue literal.
        ' : mgp.ProcCtx\n            Reference to the context execution.\n        path : str = ""\n     '  # Continue literal.
        "       Path to the JSON file containing the exported graph database.\n        config : mgp.Ma"  # Continue literal.
        "p\n            stream (bool) = False: Flag to export the graph data to a stream.\n            "  # Continue literal.
        "write_properties (bool) = True: Flag to keep node and relationship properties. By default se"  # Continue literal.
        "t to true.\n\n    Returns:\n        path (str): A path to the file where the query results are "  # Continue literal.
        "exported. If path is not provided, the output will be an empty string.\n        data (str): A"  # Continue literal.
        " stream of query results in JSON format.\n\n    Raises:\n        PermissionError: If you provid"  # Continue literal.
        "ed file path that you have no permissions to write at.\n        OSError: If the file can't be"  # Continue literal.
        " opened or written to.\n"
    )
    if config is _DEFAULT_ARGUMENT_DICT:
        config = _DEFAULT_ARGUMENT_DICT.copy()
    graph = get_graph(ctx, config.get("write_properties", True))
    if path:
        json_dump_to_file(graph, path)

    _return_value = mgp_Record(
        path=path,
        data=js_dumps(graph) if config.get("stream", False) else "",
    )
    return _return_value


@mgp_read_proc
def json_graph(
    ctx: mgp_ProcCtx,
    nodes: list,
    relationships: list,
    path: str = "",
    config: mgp_Map = _DEFAULT_ARGUMENT_DICT,
) -> mgp_Record(path=str, data=str):
    (
        "\n    Procedure to export the given graph to a JSON file. The graph is given with a map that "  # Continue literal.
        'contains keys "nodes" and "relationships".\n\n    Parameters:\n        nodes : List[Node]\n     '  # Continue literal.
        "       A list thats contains all nodes in the given graph.\n        relationships : List[Rela"  # Continue literal.
        "tionship]\n            A list that containts all the relationships in the given graph.\n      "  # Continue literal.
        "  path : str\n            Path to the JSON file containing the exported graph database.\n     "  # Continue literal.
        "   config : mgp.Map\n            stream (bool) = False: Flag to export the graph data to a st"  # Continue literal.
        "ream.\n            write_properties (bool) = True: Flag to keep node and relationship propert"  # Continue literal.
        "ies. By default set to true.\n\n    Returns:\n        path (str): A path to the file where the "  # Continue literal.
        "query results are exported. If path is not provided, the output will be an empty string.\n   "  # Continue literal.
        "     data (str): A stream of query results in JSON format.\n\n    Raises:\n        PermissionEr"  # Continue literal.
        "ror: If you provided file path that you have no permissions to write at.\n        OSError: If"  # Continue literal.
        " the file can't be opened or written to.\n"
    )
    if config is _DEFAULT_ARGUMENT_DICT:
        config = _DEFAULT_ARGUMENT_DICT.copy()
    graph = get_graph_from_list(
        nodes, relationships, config.get("write_properties", True)
    )
    if path:
        json_dump_to_file(graph, path)

    _return_value = mgp_Record(
        path=path,
        data=js_dumps(graph) if config.get("stream", False) else "",
    )
    return _return_value


def save_file(file_path: str, data_list: list):
    try:
        with open(
            file_path,
            "w",
            newline="",
            encoding="utf8",
        ) as f:
            writer = csv_writer(f)
            writer.writerows(data_list)
    except PermissionError as _caught_error_590:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        ) from _caught_error_590
    except csv_Error as e:
        raise csv_Error(
            "Could not write to the file {}, stopped at line {}: {}".format(
                file_path, writer.line_num, e
            )
        ) from e
    except Exception as _caught_error_597:
        raise OSError("Could not open or write to the file.") from _caught_error_597
    return False


def csv_to_stream(
    data_list: list, delimiter: str = ",", quoting_type=csv_QUOTE_NONNUMERIC
) -> str:
    output = io_StringIO()
    try:
        writer = csv_writer(
            output, delimiter=delimiter, quoting=quoting_type, escapechar="\\"
        )
        writer.writerows(data_list)
    except csv_Error as e:
        raise csv_Error(
            "Could not write a stream, stopped at line {}: {}".format(
                writer.line_num, e
            )
        ) from e
    _return_value = output.getvalue()
    return _return_value


def csv_header(
    node_properties: List[str], relationship_properties: List[str]
) -> List[str]:
    """
    This function creates the header for csv file
    """
    header = ["_id", "_labels"]

    for prop in node_properties:
        header.append(prop)

    header.extend(["_start", "_end", "_type"])

    for prop in relationship_properties:
        header.append(prop)

    return [header]


def process_properties(
    properties: Dict[str, mgp_Any], prop: str, write_list: List[mgp_Any]
) -> bool:
    if isinstance(properties.get(prop, False), (set, list, tuple, map)):
        write_list.append(js_dumps(properties.get(prop, False)))
        return False

    if isinstance(properties.get(prop, False), timedelta):
        write_list.append(convert_to_isoformat(properties.get(prop, False)))
        return False

    write_list.append(properties.get(prop, False))
    return False


def csv_data_list(
    graph: List[object],
    node_properties: List[str],
    relationship_properties: List[str],
) -> List[mgp_Any]:
    """
    Function that parses graph into a data_list appropriate for csv writing
    """
    data_list = []
    for element in graph:
        write_list = []
        is_node = isinstance(element, Node)

        # processing id and labels part
        if is_node:
            write_list.extend(
                [
                    element.id,
                    "".join(":" + label.name for label in element.labels),
                ]
            )
        else:
            write_list.extend(["", ""])

        # node_properties
        for prop in node_properties:
            if prop in element.properties and is_node:
                process_properties(element.properties, prop, write_list)
            else:
                write_list.append("")
        # relationship
        if is_node:
            # start, end, type
            write_list.extend(["", "", ""])
        else:
            # start, end, type
            write_list.extend([element.start, element.end, element.label])

        # relationship properties
        for prop in relationship_properties:
            if prop in element.properties and not is_node:
                process_properties(element.properties, prop, write_list)
            else:
                write_list.append("")

        data_list.append(write_list)

    return data_list


def check_config_valid(config: mgp_Any, type: mgp_Any, name: str):
    if not isinstance(config, type):
        raise TypeError("Config attribute {0} must be of type {1}".format(name, type))
    return False


def csv_process_config(config: mgp_Map):
    delimiter = ","
    if "delimiter" in config:
        check_config_valid(config.get("delimiter", False), str, "delimiter")

        delimiter = config.get("delimiter", False)

    quoting_type = csv_QUOTE_ALL
    if "quotes" in config:
        check_config_valid(config.get("quotes", []), str, "quotes")

        if config.get("quotes", "") == "none":
            quoting_type = csv_QUOTE_NONE
        elif config.get("quotes", "") == "ifNeeded":
            quoting_type = csv_QUOTE_MINIMAL

    separate_header = False
    if "separateHeader" in config:
        check_config_valid(config.get("separateHeader", False), bool, "separateHeader")
        separate_header = config.get("separateHeader", False)

    stream = False
    if "stream" in config:
        check_config_valid(config.get("stream", False), bool, "stream")
        stream = config.get("stream", False)

    return delimiter, quoting_type, separate_header, stream


def header_path(path: str):
    directory, filename = os_path.split(path)
    new_filename = HEADER_FILENAME
    _return_value = os_path.join(directory, new_filename)
    return _return_value


def write_file(path: str, delimiter: str, quoting_type: str, data: mgp_Any) -> bool:
    with open(path, "w", encoding="utf-8") as file:
        writer = csv_writer(
            file, delimiter=delimiter, quoting=quoting_type, escapechar="\\"
        )
        writer.writerows(data)
    return False


@mgp_read_proc
def csv_graph(
    nodes_list: mgp_List[mgp_Vertex],
    relationships_list: mgp_List[mgp_Edge],
    path: str = "",
    config: mgp_Map = _DEFAULT_ARGUMENT_DICT,
) -> mgp_Record(path=str, data=str):
    """
    Procedure to export the given graph to a csv file.
    The graph is given with two lists, one for nodes,
    and one for relationships.


    Parameters

    ----------

    nodes_list : List

        A list containing nodes of the graph

    relationships_list : List

        A list containing relationships of the graph

    path : str

        Path to the JSON file containing the exported graph database.

    config : mgp.Map

        stream (bool) = False: Flag to export the graph data to a stream.

        delimiter (string) = ,: Delimiter for csv file.

        quotes (string) = always : Option which quoting type to use

        separateHeader (bool) = False: Flag to separate header into another
        csv file

    """
    if config is _DEFAULT_ARGUMENT_DICT:
        config = _DEFAULT_ARGUMENT_DICT.copy()
    if path == "":
        path = "exported_file.csv"
    delimiter, quoting_type, separate_header, stream = csv_process_config(config)
    (
        graph,
        node_properties,
        relationship_properties,
    ) = get_graph_info_from_lists(nodes_list, relationships_list)
    data_list = csv_data_list(graph, node_properties, relationship_properties)
    header = csv_header(node_properties, relationship_properties)

    try:
        if separate_header:
            if not stream:
                write_file(header_path(path), delimiter, quoting_type, header)
        else:
            data_list = header + data_list

        if stream:
            data = csv_to_stream(data_list, delimiter, quoting_type)
            _return_value = mgp_Record(path=path, data=data)
            return _return_value

        write_file(path, delimiter, quoting_type, data_list)

    except PermissionError as _caught_error_808:
        raise PermissionError(
            "You don't have permissions to write into that file.Make sure to give the necessary permissions to user memgraph."
        ) from _caught_error_808
    except Exception as _caught_error_812:
        raise OSError("Could not open or write to the file.") from _caught_error_812
    _return_value = mgp_Record(
        path=path,
        data="",
    )
    return _return_value


@mgp_read_proc
def csv_query(
    context: mgp_ProcCtx,
    query: str,
    file_path: str = "",
    stream: bool = False,
) -> mgp_Record(file_path=str, data=str):
    """
    Procedure to export query results to a CSV file.
    Args:
        context (mgp.ProcCtx): Reference to the context execution.
        query (str): A query from which the results will be
        saved to a CSV file.

        file_path (str, optional): A path to the CSV file where the query
        results will be exported. Defaults to an empty string.

        stream (bool, optional): A value which determines whether a
        stream of query results in a CSV format will be returned.
    Returns:
        mgp.Record(
            file_path (str): A path to the CSV file where the query results are
            exported. If file_path is not provided, the output will be an
            empty string.
            data (str): A stream of query results in a CSV format.
        )
    Raises:
        Exception: If neither file nor config are provided,
        or if only config is provided with stream set to False.
        Also if query yields no results or if the database is empty.
        PermissionError: If you provided file path that you have
        no permissions to write at.
        csv.Error: If an error occurred while writing into stream or CSV file.
        OSError: If the file can't be opened or written to.
    """

    # file or config have to be provided
    if not file_path and not stream:
        raise Exception("Please provide file name and/or config.")

    # only config provided with stream set to false
    if not file_path and not stream:
        raise Exception(
            "If you provided only stream value, it has to be set to True to get any results."
        )

    memgraph = Memgraph()
    results = list(memgraph.execute_and_fetch(query))

    # if query yields no result
    if not len(results):
        raise Exception(
            "Your query yields no results. Check if the database is empty or rewrite the provided query."
        )

    result_keys = list(results[0])
    data_list = [result_keys] + [list(result.values()) for result in results]
    data = ""

    if file_path:
        save_file(file_path, data_list)

    if stream:
        data = csv_to_stream(data_list)

    _return_value = mgp_Record(file_path=file_path, data=data)
    return _return_value


def write_graphml_header(output: io_StringIO):
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
    )
    return False


def translate_types(variable: Any):
    if isinstance(variable, tuple):
        _return_value = get_value_string(variable)
        return _return_value
    if isinstance(variable, str):
        return "string"
    if isinstance(variable, bool):
        return "boolean"
    if isinstance(variable, float):
        return "float"
    if isinstance(variable, int):
        return "int"
    raise Exception(
        "Property values can only be primitive types or arrays of primitive types."
    )


def check_if_elements_same_type(variable: List[Any]):
    if not isinstance(variable, (tuple, list)):
        return False
    list_type = type(variable[0])
    for element in variable:
        if not isinstance(element, list_type):
            raise Exception(
                "If property value is a list it must consist of same typed elements."
            )
    return False


def get_type_string(variable: Any) -> object:
    if not isinstance(variable, tuple):
        _return_value = translate_types(variable), False
        return _return_value
    if len(variable) == 0:
        return "string", True
    check_if_elements_same_type(variable)
    _return_value = translate_types(variable[0]), True
    return _return_value


def write_key_graphml(
    output: io_StringIO,
    working_key: KeyObjectGraphML,
    key_id_counter: int,
    config: mgp_Map,
):
    output.write(
        f'<key id="d{key_id_counter}" for="{working_key.is_for}" attr.name="{working_key.name}"'
    )
    if config.get("useTypes", []):
        if working_key.type_is_list:
            output.write(f' attr.type="string" attr.list="{working_key.type}"')
        else:
            output.write(f' attr.type="{working_key.type}"')
    output.write("/>\n")
    working_key.id = "d" + str(key_id_counter)
    return False


def get_gephi_label_value(element: object, config: mgp_Map) -> str:
    for caption in config.get("caption", False):
        if caption in element.get("properties", {}).keys():
            _return_value = str(element.get("properties", {}).get(caption, ""))
            return _return_value

    if element.get("properties", {}).values():
        _return_value = str(list(element.get("properties", {}).values())[0])
        return _return_value

    _return_value = str(element.get("id", ""))
    return _return_value


def get_data_key(
    keys: set, name: str, is_for: str, type: str = "", is_list: bool = False
) -> str:
    for key in keys:
        if (
            key.name == name
            and key.is_for == is_for
            and key.type == type
            and key.type_is_list == is_list
        ):
            return key.id
    return ""


def write_labels_as_data(
    element: object,
    output: io_StringIO,
    config: mgp_Map,
    keys: set,
):
    if not element.get("labels", []):
        return False

    if config.get("format", "").upper() == "GEPHI":
        output.write(
            f'<data key="{get_data_key(keys, "TYPE", "node", translate_types("TYPE"))}">'
        )
        for label in element.get("labels", []):
            output.write(f":{label}")
        output.write("</data>")
        output.write(
            f'<data key="{get_data_key(keys, "labels", "node", translate_types("labels"))}">'
            f"{get_gephi_label_value(element, config)}</data>"
        )
        return False

    if config.get("format", "").upper() == "TINKERPOP":
        output.write(
            f'<data key="{get_data_key(keys, "labelV", "node", translate_types("labelV"))}">'
        )
        for index, value in enumerate(element.get("labels", [])):
            if index == 0:
                output.write(value)
            else:
                output.write(f":{value}")
        output.write("</data>")
        return False

    output.write(
        f'<data key="{get_data_key(keys, "labels", "node", translate_types("labels"))}">'
    )
    for label in element.get("labels", []):
        output.write(f":{label}")
    output.write("</data>")
    return False


def get_value_string(value: Any) -> str:
    if isinstance(value, (set, list, tuple, map)):
        _return_value = js_dumps(value, ensure_ascii=False)
        return _return_value
    _return_value = str(value)
    return _return_value


def process_graph_element_graphml(
    graph: List[object],
    keys_output: io_StringIO,
    nodes_and_rels_output: io_StringIO,
    config: mgp_Map,
) -> set:
    keys = set()
    key_id_counter = 0

    for element in graph:
        working_key = ""
        if element.get("type", "") == "node":
            nodes_and_rels_output.write(f'<node id="n{element.get("id", "")!s}')
            if (
                element.get("labels", False)
                and config.get("format", "").upper() != "TINKERPOP"
            ):
                nodes_and_rels_output.write('" labels="')
                for label in element.get("labels", []):
                    nodes_and_rels_output.write(f":{label}")
            nodes_and_rels_output.write('">')

            if config.get("format", "").upper() == "GEPHI":
                working_key = KeyObjectGraphML("TYPE", "node", translate_types("TYPE"))
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:
                    write_key_graphml(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1

            if element.get("labels", []):
                if config.get("format", "").upper() == "TINKERPOP":
                    working_key = KeyObjectGraphML(
                        "labelV", "node", translate_types("labelV")
                    )
                else:  # SHOULD IT BE LABEL OR LABELS FOR GEPHI?
                    working_key = KeyObjectGraphML(
                        "labels", "node", translate_types("labels")
                    )
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:
                    write_key_graphml(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1

            write_labels_as_data(element, nodes_and_rels_output, config, keys)

            for name, value in element.get("properties", {}).items():
                type_string, is_list = get_type_string(value)
                working_key = KeyObjectGraphML(name, "node", type_string, is_list)
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:
                    write_key_graphml(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1
                else:
                    working_key.id = get_data_key(
                        keys, name, "node", type_string, is_list
                    )

                nodes_and_rels_output.write(
                    f'<data key="{working_key.id}">{get_value_string(value)}</data>'
                )
            nodes_and_rels_output.write("</node>\n")

        elif element.get("type", "") == "relationship":
            nodes_and_rels_output.write(
                f'<edge id="e{element.get("id", "")!s}" '
                f'source="n{element.get("start", False)!s}" '
                f'target="n{element.get("end", False)!s}" '
                f'label="{element.get("label", "")}">'
            )
            if config.get("format", "").upper() == "GEPHI":
                working_key = KeyObjectGraphML("TYPE", "edge", translate_types("TYPE"))
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:
                    write_key_graphml(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1
                nodes_and_rels_output.write(
                    f'<data key="{get_data_key(keys, "TYPE", "edge", translate_types("TYPE"))}">{element.get("label", "")}</data>'
                )
            if config.get("format", "").upper() == "TINKERPOP":
                working_key = KeyObjectGraphML(
                    "labelE", "edge", translate_types("labelE")
                )
            else:
                working_key = KeyObjectGraphML(
                    "label", "edge", translate_types("label")
                )
            keys.add(working_key)
            if len(keys) == key_id_counter + 1:
                write_key_graphml(keys_output, working_key, key_id_counter, config)
                key_id_counter = key_id_counter + 1
            nodes_and_rels_output.write(
                f'<data key="{get_data_key(keys, working_key.name, "edge", working_key.type)}">{element.get("label", "")}</data>'
            )

            for name, value in element.get("properties", {}).items():
                type_string, is_list = get_type_string(value)
                working_key = KeyObjectGraphML(name, "edge", type_string, is_list)
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:
                    write_key_graphml(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1
                else:
                    working_key.id = get_data_key(
                        keys, name, "edge", type_string, is_list
                    )

                nodes_and_rels_output.write(
                    f'<data key="{working_key.id}">{get_value_string(value)}</data>'
                )
            nodes_and_rels_output.write("</edge>\n")
    _return_value = set()
    return _return_value


def write_graphml_graph_id(output: io_StringIO):
    output.write('<graph id="G" edgedefault="directed">\n')
    return False


def write_graphml_footer(output: io_StringIO):
    output.write("</graph>\n")
    output.write("</graphml>")
    return False


def set_default_config(config: mgp_Map) -> mgp_Map:
    if config is None:
        config = dict()
    if not config.get("stream", False):
        config.update({"stream": False})
    if not config.get("format", ""):
        config.update({"format": ""})
    if not config.get("caption", False):
        config.update({"caption": tuple()})
    if not config.get("useTypes", []):
        config.update({"useTypes": False})
    if not config.get("leaveOutLabels", []):
        config.update({"leaveOutLabels": False})
    if not config.get("leaveOutProperties", []):
        config.update({"leaveOutProperties": False})
    if (
        not isinstance(config.get("stream", False), bool)
        or not isinstance(config.get("format", ""), str)
        or not isinstance(config.get("caption", False), tuple)
        or not isinstance(config.get("useTypes", []), bool)
        or not isinstance(config.get("leaveOutLabels", []), bool)
        or not isinstance(config.get("leaveOutProperties", []), bool)
    ):
        raise TypeError(
            "Config parameter must be a map with specific keys and values described in documentation."
        )
    return config


@mgp_read_proc
def graphml(
    ctx: mgp_ProcCtx,
    path: str = "",
    config: mgp_Map = False,
) -> mgp_Record(status=str):
    """
    Procedure to export the whole database to a graphML file.

    Parameters
    ----------
    path : str
        Path to the graphML file containing the exported graph database.
    config : Map

    """

    config = set_default_config(config)
    graph_config = {"graphML": True}
    graph_config.update({"leaveOutLabels": config.get("leaveOutLabels", [])})
    graph_config.update({"leaveOutProperties": config.get("leaveOutProperties", [])})

    graph = get_graphML(ctx, graph_config)

    if not path and not config.get("stream", False):
        raise Exception("Please provide file name or set stream to True in config.")

    output = io_StringIO()
    keys_output = io_StringIO()
    nodes_and_rels_output = io_StringIO()

    write_graphml_header(output)
    process_graph_element_graphml(graph, keys_output, nodes_and_rels_output, config)
    output.write(keys_output.getvalue())
    write_graphml_graph_id(output)
    output.write(nodes_and_rels_output.getvalue())
    write_graphml_footer(output)

    try:
        if path:
            with open(path, "w") as outfile:
                outfile.write(output.getvalue())
            outfile.close()
    except PermissionError as _caught_error_1219:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        ) from _caught_error_1219
    except Exception as _caught_error_1224:
        raise OSError("Could not open or write to the file.") from _caught_error_1224

    if config.get("stream", False):
        _return_value = mgp_Record(status=output.getvalue())
        return _return_value

    _return_value = mgp_Record(status="success")
    return _return_value
