"""Utilities for import util."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from json import load as js_load
from json import loads as js_loads
from re import compile as re_compile

from defusedxml import ElementTree as ET
from gqlalchemy.memgraph_constants import MG_ENCRYPTED, MG_HOST, MG_PASSWORD, MG_PORT, MG_USERNAME
from mgclient import MG_SSLMODE_DISABLE, MG_SSLMODE_REQUIRE
from mgclient import Error as mgclient_Error
from mgclient import connect as mgclient_connect
from mgp import EdgeType as mgp_EdgeType
from mgp import Map as mgp_Map
from mgp import Nullable as mgp_Nullable
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import write_proc as mgp_write_proc

from mage.export_import_util.duration import to_duration_iso_format
from mage.export_import_util.parameters import Parameter

DEFAULT_ARGUMENT_DICT = {
    "graphML": False,
    "leaveOutLabels": False,
    "leaveOutProperties": False,
}

# Exact body of str(timedelta), which convert_to_isoformat writes inside duration(...): "[-]D day[s], H:MM:SS[.ffffff]".
TIMEDELTA_TEXT = re_compile(r"(?:(-?\d+) days?, )?(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{6}))?")

DEFAULT_CYPHER_IMPORT_CONFIG = {}

# Admitted cypher import settings. The target keys mirror gqlalchemy.Memgraph's connection arguments.
CYPHER_IMPORT_CONFIG_TYPES = {
    "host": str,
    "port": int,
    "username": str,
    "password": str,
    "encrypted": bool,
    "startLine": int,
}

# GraphML boolean data: the XML Schema lexical forms plus the exporter's str(bool) spelling.
GRAPHML_TRUE_TOKENS = {"true", "True", "1"}
GRAPHML_FALSE_TOKENS = {"false", "False", "0"}


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
        type_is_list: bool = False,
        default_value: str = "",
    ):
        self.name = name
        self.is_for = is_for
        self.type = type
        self.type_is_list = type_is_list
        self.default_value = default_value

    def __hash__(self):
        computed_return_value = hash(
            (
                self.name,
                self.is_for,
                self.type,
                self.type_is_list,
                self.default_value,
            )
        )
        return computed_return_value

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        computed_return_value = (
            self.name == other.name
            and self.is_for == other.is_for
            and self.type == other.type
            and self.type_is_list == other.type_is_list
            and self.default_value == other.default_value
        )
        return computed_return_value


def convert_to_isoformat(property: object):
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
    if isinstance(property, timedelta):
        computed_return_value = to_duration_iso_format(property)
        return computed_return_value

    if isinstance(property, (time, date, datetime)):
        computed_return_value = property.isoformat()
        return computed_return_value

    else:
        return property


def get_graph(
    ctx: mgp_ProcCtx,
    config: mgp_Map = DEFAULT_ARGUMENT_DICT,
) -> list[object]:
    """
    config : Map
        - graphML: bool
        - leaveOutLabels: bool
        - leaveOutProperties: bool

    """
    if config is DEFAULT_ARGUMENT_DICT:
        config = DEFAULT_ARGUMENT_DICT.copy()
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = []
        properties = dict()
        if not config.get("leaveOutLabels", []):
            labels = [label.name for label in vertex.labels]
        if config.get("graphML", False) and not config.get("leaveOutProperties", []):
            properties = {key: convert_to_isoformat_graphML(vertex.properties.get(key, False)) for key in vertex.properties.keys()}
        elif not config.get("leaveOutProperties", []):
            properties = {key: convert_to_isoformat(vertex.properties.get(key, False)) for key in vertex.properties.keys()}

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            if config.get("graphML", False) and not config.get("leaveOutProperties", []):
                properties = {key: convert_to_isoformat_graphML(edge.properties.get(key, False)) for key in edge.properties.keys()}
            elif not config.get("leaveOutProperties", []):
                properties = {key: convert_to_isoformat(edge.properties.get(key, False)) for key in edge.properties.keys()}

            relationships.append(
                Relationship(
                    edge.to_vertex.id,
                    edge.id,
                    edge.type.name,
                    properties,
                    edge.from_vertex.id,
                ).get_dict()
            )

    computed_return_value = nodes + relationships
    return computed_return_value


def parse_timedelta_text(text: str) -> timedelta:
    """Inverse of str(timedelta): signed days, whole seconds, zero and fractions all round-trip."""
    match = TIMEDELTA_TEXT.fullmatch(text)
    if not match:
        raise ValueError(f"not a duration body: {text!r}")
    days, hours, minutes, seconds, microseconds = (int(group) for group in match.groups(default="0"))
    value = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    return value


TEMPORAL_DECODERS = (
    (Parameter.DURATION.value, parse_timedelta_text),
    (Parameter.LOCALTIME.value, time.fromisoformat),
    (Parameter.LOCALDATETIME.value, datetime.fromisoformat),
    (Parameter.DATE.value, date.fromisoformat),
)


def convert_from_isoformat(property: object):
    """
    Decodes the typed temporal wrappers convert_to_isoformat writes. A value is decoded only when re-encoding it
    reproduces the string exactly; any other string, including one that merely starts with a wrapper name, stays an
    ordinary string.
    """
    if not isinstance(property, str) or not property.endswith(")"):
        return property
    for prefix, decode in TEMPORAL_DECODERS:
        if not property.startswith(prefix):
            continue
        try:
            value = decode(property[len(prefix) : -1])
        except ValueError:
            return property
        if convert_to_isoformat(value) != property:
            return property
        return value
    return property


def create_vertex(ctx: mgp_ProcCtx, properties: dict[str, object], labels: list[str]):
    vertex = ctx.graph.create_vertex()
    vertex_properties = vertex.properties

    for key, value in properties.items():
        vertex_properties[key] = convert_from_isoformat(value)

    for label in labels:
        vertex.add_label(label)

    return vertex.id


def create_edge(
    ctx: mgp_ProcCtx,
    properties: dict[str, object],
    start_node_id: object,
    end_node_id: object,
    type: str,
    vertex_ids: dict[object, int],
):
    # Memgraph vertex ids are non-negative, so -1 marks an endpoint this import never resolved; it must not
    # fall through to a real vertex such as id 0.
    start_vertex_id = vertex_ids.get(start_node_id, -1)
    end_vertex_id = vertex_ids.get(end_node_id, -1)
    if start_vertex_id < 0 or end_vertex_id < 0:
        endpoints = ((start_node_id, start_vertex_id), (end_node_id, end_vertex_id))
        unresolved = [node_id for node_id, vertex_id in endpoints if vertex_id < 0]
        raise KeyError(f"Relationship endpoint(s) {unresolved!r} do not identify an imported or explicitly matched node")
    vertex_from = ctx.graph.get_vertex_by_id(start_vertex_id)
    vertex_to = ctx.graph.get_vertex_by_id(end_vertex_id)
    edge = ctx.graph.create_edge(vertex_from, vertex_to, mgp_EdgeType(type))
    edge_properties = edge.properties

    for key, value in properties.items():
        edge_properties[key] = convert_from_isoformat(value)
    return False


def admit_cypher_import_config(config: mgp_Map) -> dict:
    """
    Validates the cypher import target and resume line. An omitted target is GQLAlchemy's environment-configured
    endpoint (MG_HOST, MG_PORT, MG_USERNAME, MG_PASSWORD, MG_ENCRYPT), the same one export_util.cypher_all reads.
    """
    settings = {
        "host": MG_HOST,
        "port": MG_PORT,
        "username": MG_USERNAME,
        "password": MG_PASSWORD,
        "encrypted": MG_ENCRYPTED,
        "startLine": 1,
    }
    for key, value in dict(config).items():
        expected_type = CYPHER_IMPORT_CONFIG_TYPES.get(key, False)
        if not expected_type:
            raise KeyError(f"Unknown cypher import config key {key!r}; expected one of {sorted(CYPHER_IMPORT_CONFIG_TYPES)}.")
        if not isinstance(value, expected_type) or (expected_type is int and isinstance(value, bool)):
            raise TypeError(f"Cypher import config {key!r} must be {expected_type.__name__}, received {value!r}.")
        settings[key] = value
    if not settings.get("host", ""):
        raise ValueError("Cypher import config 'host' must name the Memgraph instance that receives the statements.")
    if not 0 < settings.get("port", 0) < 65536:
        raise ValueError(f"Cypher import config 'port' must be a TCP port, received {settings.get('port', 0)!r}.")
    if settings.get("startLine", 1) < 1:
        raise ValueError(f"Cypher import config 'startLine' must be at least 1, received {settings.get('startLine', 1)!r}.")
    return settings


@mgp_write_proc
def cypher(ctx: mgp_ProcCtx, path: str, config: mgp_Map = DEFAULT_CYPHER_IMPORT_CONFIG) -> mgp_Record:
    """
    Procedure to import the one-statement-per-line Cypher created by export_util.cypher_all.
    The lab import feature should be prefered.

    A procedure cannot run Cypher inside its calling transaction, so the statements run on a separate Bolt connection
    to the configured target and each commits on its own (index, constraint and trigger statements cannot share a
    transaction with data). The calling transaction does not own or roll back those writes. If a statement fails, the
    error reports how many statements were committed and the line to resume from; the connection is always closed.

    Parameters
    ----------
    path : str
        Path to the Cypher file that is being imported.
    config : Map
        host, port, username, password, encrypted: the receiving Memgraph instance; omitted keys use GQLAlchemy's
            MG_HOST/MG_PORT/MG_USERNAME/MG_PASSWORD/MG_ENCRYPT environment configuration.
        startLine (int) = 1: first file line to execute, for resuming after a reported failure.
    """
    if config is DEFAULT_CYPHER_IMPORT_CONFIG:
        config = DEFAULT_CYPHER_IMPORT_CONFIG.copy()
    settings = admit_cypher_import_config(config)
    target = f"{settings.get('host', '')}:{settings.get('port', 0)}"
    start_line = settings.get("startLine", 1)

    try:
        with open(path, "r") as file:
            lines = file.readlines()
    except OSError as err:
        raise OSError("Could not open/read file.") from err

    try:
        connection = mgclient_connect(
            host=settings.get("host", ""),
            port=settings.get("port", 0),
            username=settings.get("username", ""),
            password=settings.get("password", ""),
            sslmode=MG_SSLMODE_REQUIRE if settings.get("encrypted", False) else MG_SSLMODE_DISABLE,
        )
    except mgclient_Error as err:
        raise ConnectionError(f"Cypher import could not connect to {target}: {err}") from err

    committed = 0
    try:
        connection.autocommit = True
        for line_number, line in enumerate(lines, start=1):
            statement = line.strip()
            if line_number < start_line or not statement:
                continue
            ctx.check_must_abort()
            cursor = connection.cursor()
            try:
                cursor.execute(statement)
                cursor.fetchall()
            except mgclient_Error as err:
                raise RuntimeError(
                    f"Cypher import into {target} failed at line {line_number}: {err}. {committed} statement(s) from line "
                    f"{start_line} were already committed and remain; fix the statement and rerun with config "
                    f"{{startLine: {line_number}}} to resume."
                ) from err
            committed += 1
    finally:
        connection.close()

    computed_return_value = mgp_Record()
    return computed_return_value


@mgp_write_proc
def json(ctx: mgp_ProcCtx, path: str) -> mgp_Record:
    """
    Procedure to import the JSON created by the export_util.json procedure.

    Parameters
    ----------
    path : str
        Path to the JSON file that is being imported.
    """
    try:
        with open(path, "r") as file:
            graph_objects = js_load(file)
    except Exception as caught_error_318:
        raise OSError("Could not open/read file.") from caught_error_318

    vertex_ids = dict()

    for graph_object in graph_objects:
        if all(
            key in graph_object
            for key in (
                Parameter.TYPE.value,
                Parameter.PROPERTIES.value,
                Parameter.ID.value,
            )
        ):
            type_value = graph_object.get(Parameter.TYPE.value, "")
            properties_value = graph_object.get(Parameter.PROPERTIES.value, {})
            id_value = graph_object.get(Parameter.ID.value, 0)
        else:
            raise KeyError(
                "Each graph object needs to have 'type', \
                 'properties' and 'id' keys."
            )

        if type_value == Parameter.NODE.value:
            if Parameter.LABELS.value in graph_object:
                labels_value = graph_object.get(Parameter.LABELS.value, [])
            else:
                raise KeyError("Each node object needs to have 'labels' key.")

            if id_value in vertex_ids:
                raise ValueError(f"Node id {id_value!r} appears more than once, so relationships to it are ambiguous.")
            vertex_ids[id_value] = create_vertex(ctx, properties_value, labels_value)

        elif type_value == Parameter.RELATIONSHIP.value:
            if all(
                key in graph_object
                for key in (
                    Parameter.START.value,
                    Parameter.END.value,
                    Parameter.LABEL.value,
                )
            ):
                start_node_id = graph_object.get(Parameter.START.value, 0)
                end_node_id = graph_object.get(Parameter.END.value, 0)
                edge_type = graph_object.get(Parameter.LABEL.value, "")
            else:
                raise KeyError(
                    "Each relationship object needs to have 'start', \
                     'end' and 'label' keys."
                )

            create_edge(
                ctx,
                properties_value,
                start_node_id,
                end_node_id,
                edge_type,
                vertex_ids,
            )
        else:
            raise KeyError("The provided file does not match the correct JSON format.")

    computed_return_value = mgp_Record()
    return computed_return_value


def find_node(ctx: mgp_ProcCtx, label: str, prop_key: str, prop_value: object) -> int:
    """Returns the id of the one existing vertex with `label` whose `prop_key` renders as `prop_value`."""
    matches = [
        vertex.id
        for vertex in ctx.graph.vertices
        if label in [vertex_label.name for vertex_label in vertex.labels]
        and prop_key in vertex.properties.keys()
        and str(convert_to_isoformat_graphML(vertex.properties.get(prop_key, False))) == prop_value
    ]
    if len(matches) != 1:
        raise KeyError(
            f"GraphML edge endpoint {prop_value!r} matches {len(matches)} vertices labelled {label!r} by property "
            f"{prop_key!r}; exactly one is required."
        )
    vertex_id = matches[0]
    return vertex_id


def resolve_graphml_endpoint(ctx: mgp_ProcCtx, endpoint: str, lookup: dict, real_ids: dict) -> int:
    """
    Resolves a GraphML edge endpoint to a Memgraph vertex id: a node of this document, else the one existing vertex
    the configured source/target lookup selects, else (no lookup configured) an explicitly written internal vertex id.
    """
    vertex_id = real_ids.get(endpoint, -1)
    if vertex_id >= 0:
        return vertex_id
    if lookup:
        vertex_id = find_node(ctx, lookup.get("label", ""), lookup.get("id", "id"), endpoint)
    elif endpoint.isascii() and endpoint.isdigit():
        vertex_id = int(endpoint)
    else:
        raise KeyError(f"GraphML edge endpoint {endpoint!r} is neither a node of this document nor an internal vertex id.")
    real_ids[endpoint] = vertex_id
    return vertex_id


def cast_element(text: str, type: str) -> object:
    if text == "":
        return ""
    if type == "string":
        computed_return_value = str(text)
        return computed_return_value
    if type == "int" or type == "long":
        computed_return_value = int(text)
        return computed_return_value
    if type == "boolean":
        token = text.strip()
        if token in GRAPHML_TRUE_TOKENS:
            return True
        if token in GRAPHML_FALSE_TOKENS:
            return False
        raise ValueError(f"GraphML boolean data must be true, false, 1 or 0, received {text!r}")
    if type == "float" or type == "double":
        computed_return_value = float(text)
        return computed_return_value
    if type == "":
        return text
    return False


def cast_list_member(value: object, type: str) -> object:
    """Admits one decoded JSON array member against the key's declared GraphML element type."""
    if type in ("string", "") and isinstance(value, str):
        return value
    if type == "boolean" and isinstance(value, bool):
        return value
    if type in ("int", "long") and isinstance(value, int) and not isinstance(value, bool):
        return value
    if type in ("float", "double") and isinstance(value, (int, float)) and not isinstance(value, bool):
        member = float(value)
        return member
    if type == "" and isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"GraphML list member {value!r} does not match the declared element type {type!r}")


def cast(text: str, type: str, is_list: bool) -> object:
    if is_list:
        # export_util writes list data as a JSON array (get_value_string), so it is decoded as JSON, not Python.
        try:
            values = js_loads(text)
        except (ValueError, RecursionError) as err:
            raise ValueError(f"GraphML list data is not a JSON array: {text!r}") from err
        if not isinstance(values, list):
            raise ValueError(f"GraphML list data is not a JSON array: {text!r}")
        casted_list = [cast_list_member(value, type) for value in values]
        return casted_list
    computed_return_value = cast_element(text, type)
    return computed_return_value


def set_default_keys(key_dict: dict[str, KeyObjectGraphML], properties: dict[str, object], is_for: str):
    for key_object in key_dict.values():
        if key_object.default_value != "" and key_object.is_for == is_for:
            properties.update(
                {
                    key_object.name: cast(
                        key_object.default_value,
                        key_object.type,
                        key_object.type_is_list,
                    )
                }
            )
    return False


def set_default_config(config: mgp_Map) -> mgp_Map:
    if config is None:
        config = dict()
    if not config.get("readLabels", []):
        config.update({"readLabels": False})
    if not config.get("defaultRelationshipType", False):
        config.update({"defaultRelationshipType": "RELATED"})
    if not config.get("storeNodeIds", []):
        config.update({"storeNodeIds": False})
    if not config.get("source", ""):
        config.update({"source": {}})
    if not config.get("target", False):
        config.update({"target": {}})
    if (
        not isinstance(config.get("readLabels", []), bool)
        or not isinstance(config.get("defaultRelationshipType", False), str)
        or not isinstance(config.get("storeNodeIds", []), bool)
        or not isinstance(config.get("source", ""), dict)
        or not isinstance(config.get("target", False), dict)
        or (config.get("source", "") and "label" not in config.get("source", {}).keys())
        or (config.get("target", False) and "label" not in config.get("target", {}).keys())
    ):
        raise TypeError(
            "Config parameter must be a map with specific \
             keys and values described in documentation."
        )
    return config


@mgp_write_proc
def graphml(
    ctx: mgp_ProcCtx,
    path: str = "",
    config: mgp_Nullable[mgp_Map] = None,
) -> mgp_Record:
    """
    Procedure to export the whole database to a graphML file.

    Parameters
    ----------
    path : str
        Path to the graphML file containing the exported graph database.
    config : Map

    """

    config = set_default_config(config)

    try:
        tree = ET.parse(path)
    except Exception as caught_error_488:
        raise OSError("Could not open/read file.") from caught_error_488

    root = tree.getroot()
    if root is None:
        raise ValueError("GraphML document has no root element")
    graphml_ns = root.tag.split("}")[0].strip("{")
    namespace = {"graphml": graphml_ns}

    keys: dict[str, KeyObjectGraphML] = {}

    for key in root.findall(".//graphml:key", namespace):
        working_key = KeyObjectGraphML(key.attrib.get("attr.name", ""), key.attrib.get("for", ""))
        if "attr.list" in key.attrib.keys():
            working_key.type_is_list = True
            working_key.type = key.attrib.get("attr.list", "")
        elif "attr.type" in key.attrib.keys():
            working_key.type = key.attrib.get("attr.type", "")
        child = key.findall(".//graphml:default", namespace)
        if child:
            working_key.default_value = child[0].text or ""
        working_key.id = key.attrib.get("id", "")
        keys.update({key.attrib.get("id", ""): working_key})

    real_ids = dict()

    for node in root.findall(".//graphml:node", namespace):
        labels = []
        properties = dict()
        if config.get("readLabels", []):
            labels = node.attrib.get("labels", "").split(":")
            labels.pop(0)
        if config.get("storeNodeIds", []):
            properties.update({"id": node.attrib.get("id", "")})

        set_default_keys(keys, properties, "node")

        for data in node.findall("graphml:data", namespace):
            working_key = keys.get(data.attrib.get("key", ""), False)
            if not isinstance(working_key, KeyObjectGraphML):
                working_key = KeyObjectGraphML(data.attrib.get("key", ""), "node", "string")
            if config.get("readLabels", False) and working_key.name == "labels":
                new_labels = (data.text or "").split(":")
                new_labels.pop(0)
                if new_labels != labels:
                    labels = labels + new_labels
            else:
                properties.update(
                    {
                        working_key.name: cast(
                            data.text or "",
                            working_key.type,
                            working_key.type_is_list,
                        )
                    }
                )

        node_id = node.attrib.get("id", "")
        if node_id in real_ids:
            raise ValueError(f"GraphML node id {node_id!r} appears more than once, so edges to it are ambiguous.")
        real_ids[node_id] = create_vertex(ctx, properties, labels)

    for rel in root.findall(".//graphml:edge", namespace):
        if "label" in rel.attrib.keys():
            rel_type = rel.attrib.get("label", "")
        else:
            rel_type = config.get("defaultRelationshipType", False)

        properties = dict()
        set_default_keys(keys, properties, "edge")

        for data in rel.findall("graphml:data", namespace):
            working_key = keys.get(data.attrib.get("key", ""), False)
            if not isinstance(working_key, KeyObjectGraphML):
                working_key = KeyObjectGraphML(data.attrib.get("key", ""), "edge", "string")
            if not working_key.name == "label":  # Tinkerpop???
                properties.update(
                    {
                        working_key.name: cast(
                            data.text or "",
                            working_key.type,
                            working_key.type_is_list,
                        )
                    }
                )

        source = rel.attrib.get("source", "")
        target = rel.attrib.get("target", "")
        resolve_graphml_endpoint(ctx, source, config.get("source", {}), real_ids)
        resolve_graphml_endpoint(ctx, target, config.get("target", {}), real_ids)

        create_edge(
            ctx,
            properties,
            source,
            target,
            rel_type,
            real_ids,
        )

    computed_return_value = mgp_Record(status="success")
    return computed_return_value
