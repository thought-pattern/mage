"""Utilities for export util."""

from csv import (
    QUOTE_ALL as csv_QUOTE_ALL,
    QUOTE_MINIMAL as csv_QUOTE_MINIMAL,
    QUOTE_NONE as csv_QUOTE_NONE,
    QUOTE_NONNUMERIC as csv_QUOTE_NONNUMERIC,
    writer as csv_writer,
)
from datetime import date, datetime, time, timedelta
from functools import partial
from io import StringIO as io_StringIO
from itertools import chain
from json import dumps as js_dumps
from os import (
    O_RDONLY as os_O_RDONLY,
    W_OK as os_W_OK,
    access as os_access,
    chmod as os_chmod,
    close as os_close,
    fsync as os_fsync,
    open as os_open,
    path as os_path,
    remove as os_remove,
    replace as os_replace,
    stat as os_stat,
)
from stat import S_IMODE as stat_S_IMODE
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape

from gqlalchemy import Memgraph, Memgraph as gqlalchemy_Memgraph
from mgp import (
    Any as mgp_Any,
    Edge as mgp_Edge,
    List as mgp_List,
    Map as mgp_Map,
    Nullable as mgp_Nullable,
    ProcCtx as mgp_ProcCtx,
    Record as mgp_Record,
    Vertex as mgp_Vertex,
    read_proc as mgp_read_proc,
)

from mage.export_import_util.duration import to_cypher_duration
from mage.export_import_util.parameters import Parameter
from mage.export_import_util.temporal import convert_to_isoformat, convert_to_isoformat_graphML

DEFAULT_ARGUMENT_DICT = {}

HEADER_FILENAME = "header.csv"

# A Cypher string literal escapes its quote and backslash (openCypher string grammar) and its line breaks, because
# import_util.cypher reads one statement per line.
CYPHER_STRING_ESCAPES = str.maketrans({"\\": "\\\\", "'": "\\'", "\n": "\\n", "\r": "\\r"})
# A double-quoted XML attribute also escapes its quote, and the whitespace attribute-value normalization would fold.
XML_ATTRIBUTE_ESCAPES = {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"}
IMPORT_ID_PREFIX = "_IMPORT_ID_"


def cypher_identifier(name: str) -> str:
    """Backtick-quotes a label, relationship type, property key or trigger name; an embedded backtick is doubled."""
    escaped = name.replace("`", "``")
    quoted = f"`{escaped}`"
    return quoted


def xml_attribute(value: object) -> str:
    """Escapes a value for a double-quoted XML attribute."""
    escaped = xml_escape(str(value), XML_ATTRIBUTE_ESCAPES)
    return escaped


def xml_text(value: object) -> str:
    """Escapes a value for XML character data."""
    escaped = xml_escape(str(value))
    return escaped


def write_text(out, text: str) -> None:
    out.write(text)


def render_to_string(render) -> str:
    """Renders an export into memory, for the stream option whose contract is one returned string."""
    buffer = io_StringIO()
    render(buffer)
    rendered = buffer.getvalue()
    return rendered


def publish_file(path: str, render, newline: str) -> None:
    """
    Writes a complete export beside its destination and publishes it with one atomic rename, so a failed or interrupted
    export never truncates or partly replaces an artifact already at that path. The staged file is flushed and fsynced
    before the rename and the directory is fsynced after it; a staged file that is not published is removed. An existing
    destination keeps its permission bits, and one the caller may not write is refused as before.
    """
    destination = os_path.realpath(path)
    directory, name = os_path.split(destination)
    staging_path = os_path.join(directory, f".{name}.{uuid4().hex}.partial")
    published = False
    try:
        if os_path.exists(destination) and not os_access(destination, os_W_OK):
            raise PermissionError(f"Cannot write to {destination}")
        with open(staging_path, "x", encoding="utf-8", newline=newline) as staged:
            render(staged)
            staged.flush()
            os_fsync(staged.fileno())
        if os_path.exists(destination):
            os_chmod(staging_path, stat_S_IMODE(os_stat(destination).st_mode))
        os_replace(staging_path, destination)
        published = True
        directory_fd = os_open(directory, os_O_RDONLY)
        try:
            os_fsync(directory_fd)
        finally:
            os_close(directory_fd)
    except PermissionError as err:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        ) from err
    except OSError as err:
        raise OSError("Could not open or write to the file.") from err
    finally:
        if not published and os_path.exists(staging_path):
            os_remove(staging_path)


def convert_to_cypher_format(property: object) -> str:
    if isinstance(property, timedelta):
        computed_return_value = to_cypher_duration(property)
        return computed_return_value

    elif isinstance(property, time):
        computed_return_value = f"localTime('{property.isoformat()}')"
        return computed_return_value

    elif isinstance(property, datetime):
        computed_return_value = f"localDateTime('{property.isoformat()}')"
        return computed_return_value

    elif isinstance(property, date):
        computed_return_value = f"date('{property.isoformat()}')"
        return computed_return_value

    elif isinstance(property, str):
        computed_return_value = f"'{property.translate(CYPHER_STRING_ESCAPES)}'"
        return computed_return_value

    elif isinstance(property, tuple):  # list
        computed_return_value = "[" + ", ".join([convert_to_cypher_format(item) for item in property]) + "]"
        return computed_return_value

    elif isinstance(property, dict):
        computed_return_value = (
            "{" + ", ".join([f"{cypher_identifier(k)}: {convert_to_cypher_format(v)}" for k, v in property.items()]) + "}"
        )
        return computed_return_value

    computed_return_value = str(property)
    return computed_return_value


def get_properties_cypher(object, write_properties: bool) -> dict:
    computed_return_value = (
        {key: convert_to_cypher_format(value) for key, value in object.properties.items()} if write_properties else {}
    )
    return computed_return_value


def format_properties_cypher(properties) -> str:
    computed_return_value = "{" + ", ".join([f"{cypher_identifier(k)}: {v}" for k, v in properties.items()]) + "}"
    return computed_return_value


def cypher_property_list(properties) -> str:
    """Renders the n.`property` list of an index or constraint; Memgraph reports one name or a sequence of names."""
    names = [properties] if isinstance(properties, str) else list(properties)
    rendered = ", ".join(f"n.{cypher_identifier(name)}" for name in names)
    return rendered


def write_cypher_export(out, ctx: mgp_ProcCtx, memgraph: gqlalchemy_Memgraph, config: mgp_Map, import_id: str) -> None:
    """
    Writes the export one statement per line. The graph is read twice, nodes then relationships, so neither the graph
    nor the rendered script is held in memory.
    """
    if config.get("write_triggers", True):
        triggers = memgraph.execute_and_fetch("SHOW TRIGGERS;")
        for trigger in triggers:
            trigger_name = cypher_identifier(trigger.get("trigger name", ""))
            event_type = trigger.get("event type", "")
            phase = trigger.get("phase", "")
            statement = trigger.get("statement", "")
            out.write(f"CREATE TRIGGER {trigger_name} ON {event_type} {phase} EXECUTE {statement};\n")
        out.write("\n")

    if config.get("write_constraints", True):
        constraints = memgraph.execute_and_fetch("SHOW CONSTRAINT INFO;")
        for constraint in constraints:
            constraint_type = constraint.get("constraint type", "")
            label = cypher_identifier(constraint.get("label", ""))
            properties = cypher_property_list(constraint.get("properties", []))

            if constraint_type == "exists":
                out.write(f"CREATE CONSTRAINT ON (n:{label}) ASSERT EXISTS ({properties});\n")
            elif constraint_type == "unique":
                out.write(f"CREATE CONSTRAINT ON (n:{label}) ASSERT {properties} IS UNIQUE;\n")
            else:
                raise ValueError("Unknown constraint type.")
        out.write("\n")

    if config.get("write_indexes", True):
        indexes = memgraph.execute_and_fetch("SHOW INDEX INFO;")
        for index in indexes:
            index_type = index.get("index type", "")
            label = cypher_identifier(index.get("label", ""))
            if index_type == "label":
                out.write(f"CREATE INDEX ON :{label};\n")
            elif index_type == "label+property":
                properties = index.get("property", [])
                names = [properties] if isinstance(properties, str) else list(properties)
                out.write(f"CREATE INDEX ON :{label}({', '.join(cypher_identifier(name) for name in names)});\n")
            else:
                raise ValueError("Unknown index type.")
        out.write("\n")

    write_properties = config.get("write_properties", True)
    import_name = cypher_identifier(import_id)
    for vertex in ctx.graph.vertices:
        labels = "".join(f":{cypher_identifier(label.name)}" for label in vertex.labels)
        properties = get_properties_cypher(vertex, write_properties)
        properties[import_id] = f"{vertex.id}"
        out.write(f"CREATE (n{labels}:{import_name} {format_properties_cypher(properties)});\n")

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            properties_str = format_properties_cypher(get_properties_cypher(edge, write_properties))
            out.write(
                f"MATCH (n:{import_name} {{{import_name}: {edge.from_vertex.id}}}) MATCH (m:{import_name} {{{import_name}: "
                f"{edge.to_vertex.id}}}) CREATE (n)-[:{cypher_identifier(edge.type.name)} {properties_str}]->(m);\n"
            )

    out.write(f"MATCH (n:{import_name}) REMOVE n:{import_name} REMOVE n.{import_name};\n")


@mgp_read_proc
def cypher_all(
    ctx: mgp_ProcCtx,
    path: str = "",
    config: mgp_Map = DEFAULT_ARGUMENT_DICT,
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

    if config is DEFAULT_ARGUMENT_DICT:
        config = DEFAULT_ARGUMENT_DICT.copy()

    # The temporary import label/property is unique to this export, so replay never overwrites or removes a user label
    # or property of any name, and the final cleanup matches only the nodes this import created, even in a populated
    # target graph.
    import_id = f"{IMPORT_ID_PREFIX}{uuid4().hex}"
    render = partial(write_cypher_export, ctx=ctx, memgraph=gqlalchemy_Memgraph(), config=config, import_id=import_id)

    data = ""
    if config.get("stream", False):
        data = render_to_string(render)
        if path:
            publish_file(path, partial(write_text, text=data), "\n")
    elif path:
        publish_file(path, render, "\n")

    computed_return_value = mgp_Record(path=path, data=data)
    return computed_return_value


def get_properties_json(object, write_properties: bool):
    computed_return_value = (
        {key: convert_to_isoformat(value) for key, value in object.properties.items()} if write_properties else {}
    )
    return computed_return_value


# Every element generator below writes the same export element shapes, in the key order import_util.json reads:
# a node is {id, labels, properties, type: node} and a relationship {end, id, label, properties, start, type: relationship}.
def json_elements(ctx: mgp_ProcCtx, write_properties: bool):
    """Yields every node dict, then every relationship dict, reading the graph twice instead of holding it."""
    for vertex in ctx.graph.vertices:
        node = {
            Parameter.ID.value: vertex.id,
            Parameter.LABELS.value: [label.name for label in vertex.labels],
            Parameter.PROPERTIES.value: get_properties_json(vertex, write_properties),
            Parameter.TYPE.value: Parameter.NODE.value,
        }
        yield node

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            relationship = {
                Parameter.END.value: edge.to_vertex.id,
                Parameter.ID.value: edge.id,
                Parameter.LABEL.value: edge.type.name,
                Parameter.PROPERTIES.value: get_properties_json(edge, write_properties),
                Parameter.START.value: edge.from_vertex.id,
                Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
            }
            yield relationship


def graphml_elements(ctx: mgp_ProcCtx, config: mgp_Map):
    """
    Yields every node dict, then every relationship dict (two reads of the graph). Vertex and relationship property
    values use the same codec: GraphML text when config graphML is set, else the JSON/Cypher wrapper form.

    config : Map
        - graphML: bool
        - leaveOutLabels: bool
        - leaveOutProperties: bool
    """
    codec = convert_to_isoformat_graphML if config.get("graphML", False) else convert_to_isoformat
    leave_out_labels = config.get("leaveOutLabels", False)
    leave_out_properties = config.get("leaveOutProperties", False)

    for vertex in ctx.graph.vertices:
        properties = {} if leave_out_properties else {key: codec(value) for key, value in vertex.properties.items()}
        node = {
            Parameter.ID.value: vertex.id,
            Parameter.LABELS.value: [] if leave_out_labels else [label.name for label in vertex.labels],
            Parameter.PROPERTIES.value: properties,
            Parameter.TYPE.value: Parameter.NODE.value,
        }
        yield node

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            properties = {} if leave_out_properties else {key: codec(value) for key, value in edge.properties.items()}
            relationship = {
                Parameter.END.value: edge.to_vertex.id,
                Parameter.ID.value: edge.id,
                Parameter.LABEL.value: edge.type.name,
                Parameter.PROPERTIES.value: properties,
                Parameter.START.value: edge.from_vertex.id,
                Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
            }
            yield relationship


def listed_json_elements(graph_vertices: list, graph_edges: list, write_properties: bool):
    """Yields the given nodes, then the given relationships, as export dicts."""
    for vertex in graph_vertices:
        node = {
            Parameter.ID.value: vertex.id,
            Parameter.LABELS.value: [label.name for label in vertex.labels],
            Parameter.PROPERTIES.value: get_properties_json(vertex, write_properties),
            Parameter.TYPE.value: Parameter.NODE.value,
        }
        yield node

    for edge in graph_edges:
        relationship = {
            Parameter.END.value: edge.to_vertex.id,
            Parameter.ID.value: edge.id,
            Parameter.LABEL.value: edge.type.name,
            Parameter.PROPERTIES.value: get_properties_json(edge, write_properties),
            Parameter.START.value: edge.from_vertex.id,
            Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
        }
        yield relationship


def write_json_array(out, elements, indent: int) -> None:
    """
    Writes elements as one JSON array, one element at a time. A positive indent reproduces json.dump(..., indent=indent)
    for files; indent 0 reproduces json.dumps' single-line list for streams.
    """
    if indent:
        opening, separator, closing = "[\n", ",\n", "\n]"
    else:
        opening, separator, closing = "[", ", ", "]"
    written = False
    for element in elements:
        out.write(separator if written else opening)
        if indent:
            # Nest the element one level; JSON text never contains a raw newline inside a string.
            margin = " " * indent
            out.write(margin + js_dumps(element, indent=indent, default=str).replace("\n", "\n" + margin))
        else:
            out.write(js_dumps(element, default=str))
        written = True
    out.write(closing if written else "[]")


@mgp_read_proc
def json(ctx: mgp_ProcCtx, path: str = "", config: mgp_Map = DEFAULT_ARGUMENT_DICT) -> mgp_Record(path=str, data=str):
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
    if config is DEFAULT_ARGUMENT_DICT:
        config = DEFAULT_ARGUMENT_DICT.copy()
    write_properties = config.get("write_properties", True)
    if path:
        elements = json_elements(ctx, write_properties)
        publish_file(path, partial(write_json_array, elements=elements, indent=Parameter.STANDARD_INDENT.value), "\n")

    data = ""
    if config.get("stream", False):
        data = render_to_string(partial(write_json_array, elements=json_elements(ctx, write_properties), indent=0))

    computed_return_value = mgp_Record(path=path, data=data)
    return computed_return_value


@mgp_read_proc
def json_graph(
    ctx: mgp_ProcCtx,
    nodes: list,
    relationships: list,
    path: str = "",
    config: mgp_Map = DEFAULT_ARGUMENT_DICT,
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
    if config is DEFAULT_ARGUMENT_DICT:
        config = DEFAULT_ARGUMENT_DICT.copy()
    write_properties = config.get("write_properties", True)
    if path:
        elements = listed_json_elements(nodes, relationships, write_properties)
        publish_file(path, partial(write_json_array, elements=elements, indent=Parameter.STANDARD_INDENT.value), "\n")

    data = ""
    if config.get("stream", False):
        elements = listed_json_elements(nodes, relationships, write_properties)
        data = render_to_string(partial(write_json_array, elements=elements, indent=0))

    computed_return_value = mgp_Record(path=path, data=data)
    return computed_return_value


def write_csv_rows(out, rows, delimiter: str, quoting_type: mgp_Any) -> None:
    """Writes rows as the csv writer consumes them."""
    writer = csv_writer(out, delimiter=delimiter, quoting=quoting_type, escapechar="\\")
    writer.writerows(rows)


def write_query_rows(rows, writers: list) -> None:
    """Writes the first row's keys as the header, then every row, to each writer as the rows arrive."""
    row_count = 0
    for row in rows:
        if row_count == 0:
            header = list(row)
            for writer in writers:
                writer.writerow(header)
        values = list(row.values())
        for writer in writers:
            writer.writerow(values)
        row_count += 1

    if row_count == 0:
        raise Exception("Your query yields no results. Check if the database is empty or rewrite the provided query.")


def write_query_csv(out, rows, stream_writers: list) -> None:
    write_query_rows(rows, [csv_writer(out)] + stream_writers)


def csv_header(node_properties: list[str], relationship_properties: list[str]) -> list[list[str]]:
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


def csv_cell(value: object) -> object:
    """Encodes one property value as its CSV cell: collections as JSON, durations as their typed wrapper text."""
    if isinstance(value, (set, list, tuple, map)):
        cell = js_dumps(value)
    elif isinstance(value, timedelta):
        cell = convert_to_isoformat(value)
    else:
        cell = value
    return cell


def csv_property_keys(nodes_list: list[mgp_Vertex], relationships_list: list[mgp_Edge]) -> tuple[list[str], list[str]]:
    """First pass: the sorted property names of the nodes and of the relationships, which the header needs first."""
    node_properties = sorted({prop for node in nodes_list for prop in node.properties})
    relationship_properties = sorted({prop for relationship in relationships_list for prop in relationship.properties})
    return node_properties, relationship_properties


def csv_rows(
    nodes_list: list[mgp_Vertex],
    relationships_list: list[mgp_Edge],
    node_properties: list[str],
    relationship_properties: list[str],
):
    """
    Second pass: one row per node, then one per relationship, produced as the csv writer consumes them
    """
    # A property an element does not have is an empty cell.
    for node in nodes_list:
        # id and labels, the node properties, then empty start, end, type and relationship properties
        write_list = [node.id, "".join(":" + label.name for label in node.labels)]
        write_list.extend(csv_cell(node.properties.get(prop, "")) for prop in node_properties)
        write_list.extend(["", "", ""])
        write_list.extend("" for _ in relationship_properties)
        yield write_list

    for relationship in relationships_list:
        write_list: list = ["", ""]
        write_list.extend("" for _ in node_properties)
        write_list.extend([relationship.from_vertex.id, relationship.to_vertex.id, relationship.type.name])
        write_list.extend(csv_cell(relationship.properties.get(prop, "")) for prop in relationship_properties)
        yield write_list


def check_config_valid(config: mgp_Any, type: mgp_Any, name: str):
    if not isinstance(config, type):
        raise TypeError("Config attribute {0} must be of type {1}".format(name, type))
    return False


def csv_process_config(config: mgp_Map):
    delimiter = ","
    if "delimiter" in config:
        check_config_valid(config.get("delimiter", ""), str, "delimiter")

        delimiter = config.get("delimiter", "")

    quoting_type = csv_QUOTE_ALL
    if "quotes" in config:
        check_config_valid(config.get("quotes", ""), str, "quotes")

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
    computed_return_value = os_path.join(directory, new_filename)
    return computed_return_value


@mgp_read_proc
def csv_graph(
    nodes_list: mgp_List[mgp_Vertex],
    relationships_list: mgp_List[mgp_Edge],
    path: str = "",
    config: mgp_Map = DEFAULT_ARGUMENT_DICT,
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
    if config is DEFAULT_ARGUMENT_DICT:
        config = DEFAULT_ARGUMENT_DICT.copy()
    if path == "":
        path = "exported_file.csv"
    delimiter, quoting_type, separate_header, stream = csv_process_config(config)
    node_properties, relationship_properties = csv_property_keys(nodes_list, relationships_list)
    header = csv_header(node_properties, relationship_properties)

    # A separate header goes to its own file (and never into a stream); otherwise it leads the rows.
    rows = chain(
        [] if separate_header else header, csv_rows(nodes_list, relationships_list, node_properties, relationship_properties)
    )
    render = partial(write_csv_rows, rows=rows, delimiter=delimiter, quoting_type=quoting_type)

    data = ""
    if stream:
        data = render_to_string(render)
    else:
        if separate_header:
            publish_file(
                header_path(path), partial(write_csv_rows, rows=header, delimiter=delimiter, quoting_type=quoting_type), ""
            )
        publish_file(path, render, "")

    computed_return_value = mgp_Record(path=path, data=data)
    return computed_return_value


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
        raise Exception("If you provided only stream value, it has to be set to True to get any results.")

    memgraph = Memgraph()
    rows = memgraph.execute_and_fetch(query)

    # The query runs once and each row goes to the file and the stream (different CSV dialects) as it arrives, so the
    # result set is not retained for the file. A query with no rows publishes nothing.
    stream_buffer = io_StringIO()
    stream_writers = [csv_writer(stream_buffer, quoting=csv_QUOTE_NONNUMERIC, escapechar="\\")] if stream else []
    if file_path:
        publish_file(file_path, partial(write_query_csv, rows=rows, stream_writers=stream_writers), "")
    else:
        write_query_rows(rows, stream_writers)
    data = stream_buffer.getvalue()

    computed_return_value = mgp_Record(file_path=file_path, data=data)
    return computed_return_value


def write_graphml_header(output: io_StringIO):
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
    )
    return False


def translate_types(variable: object):
    if isinstance(variable, tuple):
        computed_return_value = get_value_string(variable)
        return computed_return_value
    if isinstance(variable, str):
        return "string"
    if isinstance(variable, bool):
        return "boolean"
    if isinstance(variable, float):
        return "float"
    if isinstance(variable, int):
        return "int"
    raise Exception("Property values can only be primitive types or arrays of primitive types.")


def check_if_elements_same_type(variable: mgp_Any):
    if not isinstance(variable, (tuple, list)):
        return False
    list_type = type(variable[0])
    for element in variable:
        if not isinstance(element, list_type):
            raise Exception("If property value is a list it must consist of same typed elements.")
    return False


def get_type_string(variable: mgp_Any) -> tuple[str, bool]:
    if not isinstance(variable, tuple):
        computed_return_value = translate_types(variable), False
        return computed_return_value
    if len(variable) == 0:
        return "string", True
    check_if_elements_same_type(variable)
    computed_return_value = translate_types(variable[0]), True
    return computed_return_value


def get_gephi_label_value(element: mgp_Any, config: mgp_Map) -> str:
    for caption in config.get("caption", ()):
        if caption in element.get("properties", {}).keys():
            computed_return_value = str(element.get("properties", {}).get(caption, ""))
            return computed_return_value

    if element.get("properties", {}).values():
        computed_return_value = str(list(element.get("properties", {}).values())[0])
        return computed_return_value

    computed_return_value = str(element.get("id", ""))
    return computed_return_value


def write_labels_as_data(element: mgp_Any, output: io_StringIO, config: mgp_Map, key_ids: dict) -> None:
    labels = element.get("labels", [])
    graph_format = config.get("format", "").upper()
    # An unlabelled element writes no label data in any format.
    if labels and graph_format == "GEPHI":
        output.write(f'<data key="{key_ids.get(("TYPE", "node", "string", False), "")}">')
        output.write(xml_text("".join(f":{label}" for label in labels)))
        output.write("</data>")
        output.write(
            f'<data key="{key_ids.get(("labels", "node", "string", False), "")}">'
            f"{xml_text(get_gephi_label_value(element, config))}</data>"
        )
    elif labels and graph_format == "TINKERPOP":
        output.write(f'<data key="{key_ids.get(("labelV", "node", "string", False), "")}">')
        output.write(xml_text(":".join(labels)))
        output.write("</data>")
    elif labels:
        output.write(f'<data key="{key_ids.get(("labels", "node", "string", False), "")}">')
        output.write(xml_text("".join(f":{label}" for label in labels)))
        output.write("</data>")


def get_value_string(value: object) -> str:
    if isinstance(value, (set, list, tuple, map)):
        computed_return_value = js_dumps(value, ensure_ascii=False)
        return computed_return_value
    computed_return_value = str(value)
    return computed_return_value


def graphml_element_keys(element: dict, config: mgp_Map) -> list[tuple[str, str, str, bool]]:
    """The (name, for, type, is-list) keys an element uses, in the order the writer declares them."""
    graph_format = config.get("format", "").upper()
    keys = []
    if element.get("type", "") == "node":
        is_for = "node"
        if graph_format == "GEPHI":
            keys.append(("TYPE", is_for, "string", False))
        if element.get("labels", []):
            keys.append(("labelV" if graph_format == "TINKERPOP" else "labels", is_for, "string", False))
    else:
        is_for = "edge"
        if graph_format == "GEPHI":
            keys.append(("TYPE", is_for, "string", False))
        keys.append(("labelE" if graph_format == "TINKERPOP" else "label", is_for, "string", False))

    for name, value in element.get("properties", {}).items():
        type_string, is_list = get_type_string(value)
        keys.append((name, is_for, type_string, is_list))
    return keys


def write_graphml_keys(output: io_StringIO, elements, config: mgp_Map) -> dict:
    """
    First pass: declares every key once, in first-use order, and returns each key's id for the second pass. Only the
    key schema is kept, never the graph.
    """
    key_ids = {}
    for element in elements:
        for key in graphml_element_keys(element, config):
            if key in key_ids:
                continue
            key_id = f"d{len(key_ids)}"
            key_ids[key] = key_id
            name, is_for, type_string, is_list = key
            output.write(f'<key id="{key_id}" for="{is_for}" attr.name="{xml_attribute(name)}"')
            if config.get("useTypes", False):
                if is_list:
                    output.write(f' attr.type="string" attr.list="{xml_attribute(type_string)}"')
                else:
                    output.write(f' attr.type="{xml_attribute(type_string)}"')
            output.write("/>\n")
    return key_ids


def write_graphml_elements(output: io_StringIO, elements, key_ids: dict, config: mgp_Map) -> None:
    """Second pass: writes each node and edge with its data, escaping every attribute and text value."""
    graph_format = config.get("format", "").upper()
    for element in elements:
        if element.get("type", "") == "node":
            is_for = "node"
            closing = "</node>\n"
            labels = element.get("labels", [])
            output.write(f'<node id="n{xml_attribute(element.get("id", ""))}')
            if labels and graph_format != "TINKERPOP":
                output.write(f'" labels="{xml_attribute("".join(f":{label}" for label in labels))}')
            output.write('">')
            write_labels_as_data(element, output, config, key_ids)
        else:
            is_for = "edge"
            closing = "</edge>\n"
            label = element.get("label", "")
            output.write(
                f'<edge id="e{xml_attribute(element.get("id", ""))}" '
                f'source="n{xml_attribute(element.get("start", ""))}" '
                f'target="n{xml_attribute(element.get("end", ""))}" '
                f'label="{xml_attribute(label)}">'
            )
            if graph_format == "GEPHI":
                output.write(f'<data key="{key_ids.get(("TYPE", is_for, "string", False), "")}">{xml_text(label)}</data>')
            label_key = ("labelE" if graph_format == "TINKERPOP" else "label", is_for, "string", False)
            output.write(f'<data key="{key_ids.get(label_key, "")}">{xml_text(label)}</data>')

        for name, value in element.get("properties", {}).items():
            type_string, is_list = get_type_string(value)
            key_id = key_ids.get((name, is_for, type_string, is_list), "")
            output.write(f'<data key="{key_id}">{xml_text(get_value_string(value))}</data>')
        output.write(closing)


def write_graphml(output: io_StringIO, ctx: mgp_ProcCtx, config: mgp_Map) -> None:
    """Writes the document in two reads of the graph: key declarations first, then nodes and edges."""
    graph_config = {
        "graphML": True,
        "leaveOutLabels": config.get("leaveOutLabels", False),
        "leaveOutProperties": config.get("leaveOutProperties", False),
    }
    write_graphml_header(output)
    key_ids = write_graphml_keys(output, graphml_elements(ctx, graph_config), config)
    write_graphml_graph_id(output)
    write_graphml_elements(output, graphml_elements(ctx, graph_config), key_ids, config)
    write_graphml_footer(output)


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
    if not config.get("caption", ()):
        config.update({"caption": tuple()})
    if not config.get("useTypes", False):
        config.update({"useTypes": False})
    if not config.get("leaveOutLabels", False):
        config.update({"leaveOutLabels": False})
    if not config.get("leaveOutProperties", False):
        config.update({"leaveOutProperties": False})
    if (
        not isinstance(config.get("stream", False), bool)
        or not isinstance(config.get("format", ""), str)
        or not isinstance(config.get("caption", ()), tuple)
        or not isinstance(config.get("useTypes", False), bool)
        or not isinstance(config.get("leaveOutLabels", False), bool)
        or not isinstance(config.get("leaveOutProperties", False), bool)
    ):
        raise TypeError("Config parameter must be a map with specific keys and values described in documentation.")
    return config


@mgp_read_proc
def graphml(
    ctx: mgp_ProcCtx,
    path: str = "",
    config: mgp_Nullable[mgp_Map] = None,
) -> mgp_Record(status=str):
    """
    Procedure to export the whole database to a graphML file.

    Parameters
    ----------
    path : str
        Path to the graphML file containing the exported graph database.
    config : Map

    """

    # A null (omitted) config is normalized to the defaults; a supplied map is validated.
    config = set_default_config(config)
    if not path and not config.get("stream", False):
        raise Exception("Please provide file name or set stream to True in config.")

    render = partial(write_graphml, ctx=ctx, config=config)
    if config.get("stream", False):
        data = render_to_string(render)
        if path:
            publish_file(path, partial(write_text, text=data), "\n")
        computed_return_value = mgp_Record(status=data)
        return computed_return_value

    publish_file(path, render, "\n")
    computed_return_value = mgp_Record(status="success")
    return computed_return_value
