"""Utilities for import util."""

from ast import literal_eval as ast_literal_eval
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from json import load as js_load

from defusedxml import ElementTree as ET
from gqlalchemy import Memgraph as gqlalchemy_Memgraph
from mgp import EdgeType as mgp_EdgeType
from mgp import Map as mgp_Map
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import write_proc as mgp_write_proc

from mage.export_import_util.parameters import Parameter

DEFAULT_ARGUMENT_DICT = {
    "graphML": False,
    "leaveOutLabels": False,
    "leaveOutProperties": False,
}


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


def to_duration_isoformat(value: timedelta) -> str:
    """Converts timedelta to ISO-8601 duration: P<date>T<time>"""
    date_parts: list[str] = []
    time_parts: list[str] = []

    if value.days != 0:
        date_parts.append(f"{abs(value.days)}D")

    if value.seconds != 0 or value.microseconds != 0:
        abs_seconds = abs(value.seconds)
        minutes, seconds = divmod(abs_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        microseconds = value.microseconds

        if hours > 0:
            time_parts.append(f"{hours}H")
        if minutes > 0:
            time_parts.append(f"{minutes}M")
        if seconds > 0 or microseconds > 0:
            microseconds_part = f".{abs(value.microseconds)}" if value.microseconds != 0 else ""
            time_parts.append(f"{seconds}{microseconds_part}S")

    date_duration_str = "".join(date_parts)
    time_duration_str = f"T{''.join(time_parts)}" if time_parts else ""

    computed_return_value = f"P{date_duration_str}{time_duration_str}"
    return computed_return_value


def convert_to_isoformat_graphML(property: object):
    if isinstance(property, timedelta):
        computed_return_value = to_duration_isoformat(property)
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
            if not config.get("leaveOutProperties", []):
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


def convert_from_isoformat(property: object):
    if not isinstance(property, str):
        return property

    if str.startswith(property, Parameter.DURATION.value):
        duration_iso = property.split("(")[-1].split(")")[0]
        parsed_time = datetime.strptime(duration_iso, "%H:%M:%S.%f")
        computed_return_value = timedelta(
            hours=parsed_time.hour,
            minutes=parsed_time.minute,
            seconds=parsed_time.second,
            microseconds=parsed_time.microsecond,
        )
        return computed_return_value
    elif str.startswith(property, Parameter.LOCALTIME.value):
        local_time_iso = property.split("(")[-1].split(")")[0]
        computed_return_value = time.fromisoformat(local_time_iso)
        return computed_return_value
    elif str.startswith(property, Parameter.LOCALDATETIME.value):
        local_datetime_iso = property.split("(")[-1].split(")")[0]
        computed_return_value = datetime.fromisoformat(local_datetime_iso)
        return computed_return_value
    elif str.startswith(property, Parameter.DATE.value):
        date_iso = property.split("(")[-1].split(")")[0]
        computed_return_value = date.fromisoformat(date_iso)
        return computed_return_value
    else:
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
    vertex_from = ctx.graph.get_vertex_by_id(vertex_ids.get(start_node_id, False))
    vertex_to = ctx.graph.get_vertex_by_id(vertex_ids.get(end_node_id, False))
    edge = ctx.graph.create_edge(vertex_from, vertex_to, mgp_EdgeType(type))
    edge_properties = edge.properties

    for key, value in properties.items():
        edge_properties[key] = convert_from_isoformat(value)
    return False


@mgp_write_proc
def cypher(ctx: mgp_ProcCtx, path: str) -> mgp_Record:
    """
    Procedure to import the Cypher created by the export_util.json procedure.
    The lab import feature should be prefered.

    Parameters
    ----------
    path : str
        Path to the JSON file that is being imported.
    """

    memgraph = gqlalchemy_Memgraph()
    try:
        with open(path, "r") as file:
            for query in file.readlines():
                stripped_query = query.strip()
                if stripped_query:
                    memgraph.execute(stripped_query)
    except OSError as caught_error_296:
        raise OSError("Could not open/read file.") from caught_error_296
    except Exception as caught_error_298:
        raise Exception("Unable to execute the given queries") from caught_error_298

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
    for vertex in ctx.graph.vertices:
        if (
            label in [label.name for label in vertex.labels]
            and prop_key in vertex.properties.keys()
            and str(convert_to_isoformat_graphML(vertex.properties.get(prop_key, False))) == prop_value
        ):
            return vertex.id
    return 0


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
        computed_return_value = bool(text)
        return computed_return_value
    if type == "float" or type == "double":
        computed_return_value = float(text)
        return computed_return_value
    if type == "":
        return text
    return False


def cast(text: str, type: str, is_list: bool) -> object:
    if is_list:
        casted_list = list()
        for element in ast_literal_eval(text):
            casted_list.append(cast_element(element, type))
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
    config: mgp_Map = False,
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

        real_ids.update({node.attrib.get("id", ""): create_vertex(ctx, properties, labels)})

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

        if rel.attrib.get("source", "") not in real_ids:
            if not config.get("source", ""):
                # without source/target config, we try with the internal id
                real_ids.update({rel.attrib.get("source", ""): int(rel.attrib.get("source", 0))})
            else:
                source_config = config.get("source", "")
                if "id" not in source_config.keys():
                    source_config.update({"id": "id"})
                node_id = find_node(
                    ctx,
                    source_config.get("label", ""),
                    source_config.get("id", ""),
                    rel.attrib.get("source", ""),
                )
                real_ids.update({rel.attrib.get("source", ""): node_id})

        if rel.attrib.get("target", False) not in real_ids:
            if not config.get("target", False):
                # without source/target config, we look for the internal id
                real_ids.update({rel.attrib.get("target", False): int(rel.attrib.get("target", 0))})
            else:
                target_config = config.get("target", False)
                if "id" not in target_config.keys():
                    target_config.update({"id": "id"})
                node_id = find_node(
                    ctx,
                    target_config.get("label", ""),
                    target_config.get("id", ""),
                    rel.attrib.get("target", False),
                )
                real_ids.update({rel.attrib.get("target", False): node_id})

        create_edge(
            ctx,
            properties,
            rel.attrib.get("source", ""),
            rel.attrib.get("target", False),
            rel_type,
            real_ids,
        )

    computed_return_value = mgp_Record(status="success")
    return computed_return_value
