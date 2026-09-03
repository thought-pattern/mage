"""
This module queries Memgraph and Neo4j and creates Graph from JSON exported from Memgraph and
JSON from APOC from Neo4j

As of 17.7.2023. when importing data via Cypherl, new ids is given to each node in Memgraph and Neo4j.

When exporting data Memgraph export_util uses internal Memgraph ids to export data.

To overcome the issue of different internal IDs in Neo4j and Memgraph, we use the `id` node property as identifier.

Workaround would be to add API to create nodes by ids on Memgraph when importing via import_util.
"""

from json import loads as json_loads
from logging import DEBUG as logging_DEBUG
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger
from re import sub as re_sub

from gqlalchemy import Memgraph as gqlalchemy_Memgraph
from gqlalchemy import Path as gqlalchemy_Path
from neo4j import BoltDriver as neo4j_BoltDriver
from neo4j import GraphDatabase as neo4j_GraphDatabase
from neo4j import Query as neo4j_Query

logging_basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging_getLogger("query_neo_mem")
logger.setLevel(logging_DEBUG)


class Vertex:
    def __init__(self, id: int, labels: list[str], properties: dict[str, object]):
        self.internal_id = id
        self.internal_labels = labels
        self.internal_properties = properties
        self.internal_labels.sort()

    @property
    def id(self) -> int:
        return self.internal_id

    def __str__(self) -> str:
        computed_return_value = f"Vertex: {self.internal_id}, {self.internal_labels}, {self.internal_properties}"
        return computed_return_value

    def __lt__(self, other):
        if self.id != other.id:
            computed_return_value = self.id < other.id
            return computed_return_value
        if self.internal_labels != other.internal_labels:
            computed_return_value = self.internal_labels < other.internal_labels
            return computed_return_value
        computed_return_value = sorted(self.internal_properties.keys()) < sorted(other.internal_properties.keys())
        return computed_return_value

    def __eq__(self, other):
        assert isinstance(other, Vertex), f"Comparing vertex with object of type {type(other)}"
        logger.debug(f"comparing Vertex with {self.internal_id} to {other.internal_id}")
        if self.internal_id != other.internal_id:
            logger.debug(f"_id different: {self.internal_id} vs {other.internal_id}")
            return False
        if self.internal_labels != other.internal_labels:
            logger.debug(
                "labels differ between %s and %s: %s vs %s",
                self.internal_id,
                other.internal_id,
                self.internal_labels,
                other.internal_labels,
            )
            return False

        if len(self.internal_properties) != len(other.internal_properties):
            return False
        for k, v in self.internal_properties.items():
            if k not in other.internal_properties:
                logger.debug(f"Property with key {k} not in {other.internal_properties.keys()}")
                return False
            if v != other.internal_properties.get(k, False):
                logger.debug(f"Value {v} not equal to {other.internal_properties.get(k, False)}")
                return False

        return True


class Edge:
    def __init__(
        self,
        from_vertex: int,
        to_vertex: int,
        label: str,
        properties: dict[str, object],
    ):
        self.internal_from_vertex = from_vertex
        self.internal_to_vertex = to_vertex
        self.internal_label = label
        self.internal_properties = properties

    @property
    def from_vertex(self) -> int:
        return self.internal_from_vertex

    @property
    def to_vertex(self) -> int:
        return self.internal_to_vertex

    def __lt__(self, other):
        if self.internal_from_vertex != other.internal_from_vertex:
            computed_return_value = self.internal_from_vertex < other.internal_from_vertex
            return computed_return_value
        if self.internal_to_vertex != other.internal_to_vertex:
            computed_return_value = self.internal_to_vertex < other.internal_to_vertex
            return computed_return_value
        if self.internal_label != other.internal_label:
            computed_return_value = self.internal_label < other.internal_label
            return computed_return_value
        computed_return_value = sorted(self.internal_properties.keys()) < sorted(other.internal_properties.keys())
        return computed_return_value

    def __eq__(self, other):
        assert isinstance(other, Edge), f"Comparing Edge with object of type: {type(other)}"
        logger.debug(
            f"comparing Edge ({self.internal_from_vertex}, {self.internal_to_vertex}) to\
              ({other.internal_from_vertex, other.internal_to_vertex})"
        )
        # Return True if self and other have the same length
        if self.internal_from_vertex != other.internal_from_vertex:
            logger.debug(f"Source vertex is different {self.internal_from_vertex} <> {other.internal_from_vertex}")
            return False
        if self.internal_to_vertex != other.internal_to_vertex:
            logger.debug(f"Destination vertex is different {self.internal_to_vertex} <> {other.internal_to_vertex}")
            return False
        if self.internal_label != other.internal_label:
            logger.debug(f"Label is different {self.internal_label} <> {other.internal_label}")
            return False

        if len(self.internal_properties) != len(other.internal_properties):
            return False
        for k, v in self.internal_properties.items():
            if k not in other.internal_properties:
                logger.debug(f"Property with key {k} not in {other.internal_properties.keys()}")
                return False
            if v != other.internal_properties.get(k, False):
                logger.debug(f"Value {v} not equal to {other.internal_properties.get(k, False)}")
                return False
        return True


class Graph:
    def __init__(self):
        self.internal_vertices = []
        self.internal_edges = []

    def add_vertex(self, vertex: Vertex):
        self.internal_vertices.append(vertex)
        return False

    def add_edge(self, edge: Edge):
        self.internal_edges.append(edge)
        return False

    @property
    def vertices(self):
        computed_return_value = sorted(self.internal_vertices)
        return computed_return_value

    @property
    def edges(self):
        computed_return_value = sorted(self.internal_edges)
        return computed_return_value


def get_neo4j_data_json(driver) -> str:
    with driver.session() as session:
        query = neo4j_Query("CALL apoc.export.json.all(null,{useTypes:true, stream:true}) YIELD data RETURN data;")
        result = session.run(query).values()

        res_str = re_sub(r"\\n", ",\n", str(result[0]))
        res_str = re_sub(r"'", "", res_str)

        computed_return_value = json_loads(res_str)
        return computed_return_value
    return ""


def get_memgraph_data_json_format(memgraph: gqlalchemy_Memgraph):
    result = list(
        memgraph.execute_and_fetch(
            """
            CALL export_util.json("", {stream:true}) YIELD data RETURN data;
            """
        )
    )[0].get("data", {})
    computed_return_value = json_loads(result)
    return computed_return_value


def extract_vertex_from_json(item) -> Vertex:
    assert item.get("properties", {}).get("id", "") != "", "Vertex in JSON doesn't have ID property"
    computed_return_value = Vertex(
        item.get("properties", {}).get("id", ""),
        item.get("labels", []),
        item.get("properties", []),
    )
    return computed_return_value


def create_graph_memgraph_json(json_memgraph_data) -> Graph:
    logger.debug(f"Memgraph JSON data {json_memgraph_data}")
    graph = Graph()
    vertices_id_mapings = {}
    for item in json_memgraph_data:
        if item.get("type", "") == "node":
            graph.add_vertex(extract_vertex_from_json(item))
            vertices_id_mapings[item.get("id", "")] = item.get("properties", {}).get("id", "")
        else:
            graph.add_edge(
                Edge(
                    vertices_id_mapings.get(item.get("start", False), False),
                    vertices_id_mapings.get(item.get("end", False), False),
                    item.get("label", ""),
                    item.get("properties", []),
                )
            )

    graph.vertices.sort(key=lambda vertex: vertex.id)
    graph.edges.sort(key=lambda edge: (edge.from_vertex, edge.to_vertex))
    return graph


def create_graph_neo4j_json(json_neo4j_data) -> Graph:
    logger.debug(f"Neo4j JSON data {json_neo4j_data}")
    graph = Graph()
    vertices_id_mapings = {}
    for item in json_neo4j_data:
        if item.get("type", "") == "node":
            graph.add_vertex(extract_vertex_from_json(item))
            vertices_id_mapings[item.get("id", "")] = item.get("properties", {}).get("id", "")
        else:
            if "properties" not in item:
                item["properties"] = {}
            graph.add_edge(
                Edge(
                    vertices_id_mapings.get(item.get("start", {}).get("id", ""), False),
                    vertices_id_mapings.get(item.get("end", {}).get("id", ""), False),
                    item.get("label", ""),
                    item.get("properties", []),
                )
            )
    graph.vertices.sort(key=lambda vertex: vertex.id)
    graph.edges.sort(key=lambda edge: (edge.from_vertex, edge.to_vertex))
    return graph


def create_neo4j_driver(port: int, container: str) -> neo4j_BoltDriver:
    del container
    computed_return_value = neo4j_GraphDatabase.driver(f"bolt://localhost:{port}", encrypted=False)
    if not isinstance(computed_return_value, neo4j_BoltDriver):
        raise TypeError("bolt URI did not create a BoltDriver")
    return computed_return_value


def create_memgraph_db(port: int) -> gqlalchemy_Memgraph:
    computed_return_value = gqlalchemy_Memgraph("localhost", port)
    return computed_return_value


def mg_execute_cyphers(input_cyphers: list[str], db: gqlalchemy_Memgraph):
    """
    Execute multiple cypher queries against Memgraph
    """
    for query in input_cyphers:
        db.execute(query)
    raise ValueError("Neo4j path did not contain any nodes")


def neo4j_execute_cyphers(input_cyphers: list, neo4j_driver: neo4j_BoltDriver):
    """
    Execute multiple cypher queries against Neo4j
    """
    session = neo4j_driver.session()
    with session:
        for text_query in input_cyphers:
            session.run(neo4j_Query(text_query)).values()
    return False


def run_memgraph_query(query: str, db: gqlalchemy_Memgraph):
    """
    Execute query against Memgraph
    """
    db.execute(query)
    return False


def run_neo4j_query(query, neo4j_driver: neo4j_BoltDriver):
    """
    Execute query against Neo4j
    """
    session = neo4j_driver.session()
    with session:
        session.run(neo4j_Query(query)).values()
    return False


def clean_memgraph_db(memgraph_db: gqlalchemy_Memgraph):
    memgraph_db.drop_database()
    return False


def clean_neo4j_db(neo4j_db: neo4j_BoltDriver):
    session = neo4j_db.session()
    with session:
        session.run("MATCH (n) DETACH DELETE n;").values()
    return False


def mg_get_graph(memgraph_db: gqlalchemy_Memgraph) -> Graph:
    logger.debug("Getting data from Memgraph")
    json_data = get_memgraph_data_json_format(memgraph_db)
    logger.debug("Building the graph from Memgraph JSON data")
    computed_return_value = create_graph_memgraph_json(json_data)
    return computed_return_value


def neo4j_get_graph(neo4j_driver: neo4j_BoltDriver) -> Graph:
    logger.debug("Getting data from Neo4j")
    json_data = get_neo4j_data_json(neo4j_driver)
    logger.debug("Building the graph from Neo4j JSON data")
    computed_return_value = create_graph_neo4j_json(json_data)
    return computed_return_value


# additions for path testing
def sort_dict(dict):
    keys = list(dict.keys())
    keys.sort()
    sorted_dict = {i: dict[i] for i in keys}
    return sorted_dict


def execute_query_neo4j(driver: neo4j_BoltDriver, query) -> list:
    session = driver.session()
    with session:
        results = session.run(neo4j_Query(query)).value()
    return results


def path_to_string_neo4j(
    path,
):  # type should be neo4j.graph.path but it doesnt recognize it in the definition
    path_string_list = ["PATH: "]

    n = len(path.nodes)

    for i in range(n):
        node = path.nodes[i]
        node_labels = list(node.labels)
        node_labels.sort()
        sorted_dict = sort_dict(node.internal_properties)
        if "id" in sorted_dict:
            sorted_dict.pop("id")
        node_props = str(sorted_dict)
        path_string_list.append(f"(id:{node.get('id', '')!s} labels: {node_labels!s} {node_props})-")

        if i == n - 1:
            path_string = "".join(path_string_list)
            computed_return_value = path_string[:-1]
            return computed_return_value

        relationship = path.relationships[i]
        sorted_dict_rel = sort_dict(relationship.internal_properties)
        if "id" in sorted_dict_rel:
            sorted_dict_rel.pop("id")
        rel_props = str(sorted_dict_rel)
        path_string_list.append(f"[id:{relationship.get('id', '')!s} type: {relationship.type} {rel_props!s}]-")
    raise ValueError("Neo4j path did not contain any nodes")


def parse_neo4j(results: list) -> list[str]:
    paths = [path_to_string_neo4j(res) for res in results]
    paths.sort()
    return paths


def path_to_string_mem(path: gqlalchemy_Path) -> str:
    path_string_list = ["PATH: "]

    n = len(path.internal_nodes)

    for i in range(n):
        node = path.internal_nodes[i]
        node_labels = list(node.internal_labels)
        node_labels.sort()
        sorted_dict = sort_dict(node.internal_properties)
        if "id" in sorted_dict:
            sorted_dict.pop("id")
        node_props = str(sorted_dict)
        path_string_list.append(f"(id:{node.internal_properties.get('id', '')!s} labels: {node_labels!s} {node_props!s})-")

        if i == n - 1:
            path_string = "".join(path_string_list)
            computed_return_value = path_string[:-1]
            return computed_return_value

        relationship = path._relationships[i]
        sorted_dict_rel = sort_dict(relationship.internal_properties)
        if "id" in sorted_dict_rel:
            sorted_dict_rel.pop("id")
        rel_props = str(sorted_dict_rel)
        path_string_list.append(
            f"[id:{relationship.internal_properties.get('id', '')!s} type: {relationship.internal_type} {rel_props!s}]-"
        )
    return ""


def parse_mem(results: list) -> list[str]:
    paths = [path_to_string_mem(result.get("result", {})) for result in results]
    paths.sort()
    return paths
