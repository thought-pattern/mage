"""Utilities for elastic search serialization."""

from collections.abc import Iterator
from datetime import datetime
from datetime import timezone as datetime_timezone
from itertools import batched
from json import loads as json_loads
from threading import Lock

from elasticsearch import Elasticsearch as elasticsearch_Elasticsearch
from elasticsearch import helpers as elasticsearch_helpers
from elasticsearch.helpers import parallel_bulk, streaming_bulk
from mgp import Any as mgp_Any
from mgp import Edge as mgp_Edge
from mgp import List as mgp_List
from mgp import Logger as mgp_Logger
from mgp import Map as mgp_Map
from mgp import Nullable as mgp_Nullable
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc

# Elasticsearch constants
ACTION = "action"
INDEX = "index"
ID = "_id"
SOURCE = "source"
INTERNAL_SOURCE = "_source"
SETTINGS = "settings"
NUMBER_OF_SHARDS = "number_of_shards"
NUMBER_OF_REPLICAS = "number_of_replicas"
MAPPINGS = "mappings"
DYNAMIC_TEMPLATES = "dynamic_templates"
MAPPING = "mapping"
ANALYZER = "analyzer"
STRING = "string"
EVENT_TYPE = "event_type"
CREATED_VERTEX = "created_vertex"
CREATED_EDGE = "created_edge"
VERTEX = "vertex"
EDGE = "edge"
INDEX_TYPE = "index_type"
AGGREGATIONS = "aggregations"
HITS = "hits"
TOTAL = "total"
STATUS = "status"
ERROR = "error"
REINDEX_OP_TYPES = ("index", "create")

# Bulk receipt fields
ATTEMPTED = "attempted"
INDEXED = "indexed"
REJECTED = "rejected"

# Constants
MEM_TYPE = "mem_type"
MEM_STRING = "_meme_string"
MEM_NUMBER = "_meme_number"
MEM_BOOLEAN = "_meme_boolean"
MEM_DATE = "_meme_date"
MEM_CATEGORIES_HAS_RAW = "mem_categories_has_raw"
MEM_TYPE_HAS_RAW = "mem_type_has_raw"
MEM_CATEGORIES = "mem_categories"

# Mappings of our data types
meme_mapping: dict[type, str] = {}
meme_mapping[str] = MEM_STRING
meme_mapping[int] = MEM_NUMBER
meme_mapping[float] = MEM_NUMBER
meme_mapping[bool] = MEM_BOOLEAN
meme_mapping[datetime] = MEM_DATE


# Create global logger object
logger: mgp_Logger = mgp_Logger()

# Connected client, published by connect only after it validates; the lock keeps concurrent reconnects coherent.
CLIENT = "client"
connection: dict[str, elasticsearch_Elasticsearch] = {}
connection_lock = Lock()


def connected_client() -> elasticsearch_Elasticsearch:
    # The registry holds at most the one published client, so an empty registry means connect has not succeeded.
    with connection_lock:
        active_clients = list(connection.values())
    if not active_clients:
        raise RuntimeError("Elasticsearch is not connected; call elastic_search_serialization.connect first.")
    active_client = active_clients[0]
    return active_client


# Helper method
def serialize_vertex(vertex: mgp_Vertex) -> dict[str, object]:
    """Serializes vertex to specified ElasticSearch schema.
    Args:
        vertex (mgp.Vertex): Reference to the vertex in Memgraph DB
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    source = serialize_properties(vertex.properties.items())
    source[MEM_CATEGORIES] = [label.name for label in vertex.labels]
    # The bulk helpers take document identity from the top-level _id action metadata and index _source as the body,
    # so replaying the same vertex replaces its document instead of adding an automatically identified copy.
    document = {ID: f"{vertex.id}", INTERNAL_SOURCE: source}
    return document


def serialize_edge(edge: mgp_Edge) -> dict[str, object]:
    """Serializes edge to specified ElasticSearch schema.
    Args:
        edge (mgp.Edge): Reference to the edge in Memgraph DB.
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    source = serialize_properties(edge.properties.items())
    source[MEM_TYPE] = edge.type.name
    document = {ID: f"{edge.from_vertex.id}-{edge.id}", INTERNAL_SOURCE: source}
    return document


def serialize_properties(properties: mgp_Any) -> dict[str, object]:
    """The method used to serialize properties of vertices and relationships.
    Args:
        properties (Dict[str, Any]]): Properties of nodes and relationships.
    Returns:
        Dict[str, Any]: Object that conforms ElasticSearch's schema.
    """
    source: dict[str, object] = {}
    for prop_key, prop_value in properties:
        if isinstance(prop_value, datetime):
            # A naive graph datetime is UTC wall time; an aware one is normalized to UTC first so the single Z suffix
            # (Zulu, added manually because isoformat has no option for it) states its real offset. Microseconds are dropped.
            if prop_value.tzinfo is not None:
                prop_value = prop_value.astimezone(datetime_timezone.utc).replace(tzinfo=None)
            prop_value = f"{prop_value.replace(microsecond=0).isoformat()}Z"
            source[f"{prop_key}{MEM_DATE}"] = prop_value
        elif type(prop_value) in meme_mapping:
            source[f"{prop_key}{meme_mapping.get(type(prop_value), '')}"] = prop_value
    return source


def triggered_vertex_documents(context_objects: list[mgp_Any]) -> Iterator[dict[str, object]]:
    """Yields vertex documents for the created-vertex events sent by a create trigger."""
    for context_object in context_objects:
        if context_object.get(EVENT_TYPE, "") == CREATED_VERTEX:
            # Memgraph's created_vertex event always carries its vertex; a malformed event is refused, never skipped.
            if VERTEX not in context_object:
                raise ValueError("A created_vertex trigger event carries no vertex.")
            yield serialize_vertex(context_object.get(VERTEX, False))


def triggered_edge_documents(context_objects: list[mgp_Any]) -> Iterator[dict[str, object]]:
    """Yields edge documents for the created-edge events sent by a create trigger."""
    for context_object in context_objects:
        if context_object.get(EVENT_TYPE, "") == CREATED_EDGE:
            # Memgraph's created_edge event always carries its edge; a malformed event is refused, never skipped.
            if EDGE not in context_object:
                raise ValueError("A created_edge trigger event carries no edge.")
            yield serialize_edge(context_object.get(EDGE, False))


def database_vertex_documents(context: mgp_ProcCtx) -> Iterator[dict[str, object]]:
    """Yields one document per database vertex as the bulk helper consumes them, so memory follows the chunk settings."""
    for vertex in context.graph.vertices:
        yield serialize_vertex(vertex)


def database_edge_documents(context: mgp_ProcCtx) -> Iterator[dict[str, object]]:
    """Yields one document per database relationship (each counted once, from its source vertex)."""
    for vertex in context.graph.vertices:
        for edge in vertex.out_edges:
            yield serialize_edge(edge)


def counted_documents(documents: Iterator[dict[str, object]], receipt: dict[str, object]) -> Iterator[dict[str, object]]:
    """Passes documents through to a bulk helper while counting how many it consumed."""
    for document in documents:
        receipt[ATTEMPTED] = receipt.get(ATTEMPTED, 0) + 1
        yield document


def record_bulk_results(results: Iterator[tuple[bool, dict[str, object]]], receipt: dict[str, object]) -> None:
    """Records every rejected bulk item with its server-reported identity, status and reason."""
    rejected = receipt.get(REJECTED, [])
    for ok, item in results:
        if ok:
            continue
        # Each item maps its single operation type to the server's per-document result.
        for details in item.values():
            rejected.append({ID: details.get(ID, ""), STATUS: details.get(STATUS, 0), ERROR: details.get(ERROR, "")})


def elastic_search_streaming_bulk(
    objects: Iterator[dict[str, object]],
    index: str,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: float = 2.0,
    max_backoff: float = 600.0,
    yield_ok: bool = True,
) -> dict[str, object]:
    (
        "\n    Sends streaming_bulk requests for the given objects to the provided index with the para"  # Continue literal.
        "meters specified.\n    Args:\n        objects (List[Any]): serialized nodes and edges that wil"  # Continue literal.
        "l be sent to the ElasticSearch.\n        index (str): The name of the index where you want to"  # Continue literal.
        " save the data.\n        chunk_size (int): The number of docs in one chunk sent to es (defaul"  # Continue literal.
        "t: 500).\n        max_chunk_bytes (int): The maximum size of the request in bytes (default: 1"  # Continue literal.
        "00MB).\n        raise_on_error (bool): Raise bulkIndexError containing errors (as .errors) fr"  # Continue literal.
        "om the execution of the last chunk when some occur. By default we raise.\n        raise_on_ex"  # Continue literal.
        "ception (bool): If False then don’t propagate exceptions from call to bulk and just report t"  # Continue literal.
        "he items that failed as failed.\n        max_retries (int): Maximum number of times a documen"  # Continue literal.
        "t will be retried when 429 is received, set to 0 (default) for no retries on 429.\n        in"  # Continue literal.
        "itial_backoff (float): The number of seconds we should wait before the first retry. Any subs"  # Continue literal.
        "equent retries will be powers of initial_backoff * 2**retry_number.\n        max_backoff (flo"  # Continue literal.
        "at): The maximum number of seconds a retry will wait.\n        yield_ok (float): If set to Fa"  # Continue literal.
        "lse will skip successful documents in the output.\n    Returns:\n        Dict[str, Any]: Receipt"  # Continue literal.
        " with attempted and indexed counts and every rejected item.\n"
    )
    receipt: dict[str, object] = {ATTEMPTED: 0, REJECTED: []}
    results = streaming_bulk(
        client=connected_client(),
        index=index,
        actions=counted_documents(objects, receipt),
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        yield_ok=yield_ok,
        raise_on_error=raise_on_error,
        raise_on_exception=raise_on_exception,
        max_retries=max_retries,
    )
    record_bulk_results(results, receipt)
    receipt[INDEXED] = receipt.get(ATTEMPTED, 0) - len(receipt.get(REJECTED, []))
    return receipt


def elastic_search_parallel_bulk(
    objects: Iterator[dict[str, object]],
    index: str,
    thread_count: int = 8,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    queue_size: int = 4,
) -> dict[str, object]:
    (
        "\n    Sends parallel_bulk requests for the given objects to the provided index with the param"  # Continue literal.
        "eters specified.\n    Args:\n        objects (List[Any]): Serialized nodes and edges that will"  # Continue literal.
        " be sent to the ElasticSearch.\n        index (str): The name of the index where you want to "  # Continue literal.
        "save the data.\n        thread_count (int): Size of the threadpool to use for the bulk reques"  # Continue literal.
        "ts.\n        chunk_size (int): The number of docs in one chunk sent to es (default: 500).\n   "  # Continue literal.
        "     max_chunk_bytes (int): The maximum size of the request in bytes (default: 100MB).\n     "  # Continue literal.
        "   raise_on_error (bool): Raise bulkIndexError containing errors (as .errors) from the execu"  # Continue literal.
        "tion of the last chunk when some occur. By default we raise.\n        raise_on_exception (boo"  # Continue literal.
        "l): If False then don’t propagate exceptions from call to bulk and just report the items tha"  # Continue literal.
        "t failed as failed.\n        queue_size (int): Size of the task queue between the main thread"  # Continue literal.
        " (producing chunks to send) and the processing threads.\n    Returns:\n        Dict[str, Any]: Rec"  # Continue literal.
        "eipt with attempted and indexed counts and every rejected item.\n"
    )
    receipt: dict[str, object] = {ATTEMPTED: 0, REJECTED: []}
    active_client = connected_client()
    # parallel_bulk pulls its actions on a pool thread, so graph objects are serialized here on the procedure thread one
    # window at a time. A window matches what parallel_bulk keeps in flight: one chunk per worker plus its task queue,
    # which holds max(queue_size, thread_count) chunks.
    window_size = (thread_count + max(queue_size, thread_count)) * chunk_size
    for window in batched(counted_documents(objects, receipt), window_size):
        results = parallel_bulk(
            client=active_client,
            index=index,
            actions=window,
            thread_count=thread_count,
            chunk_size=chunk_size,
            max_chunk_bytes=max_chunk_bytes,
            raise_on_error=raise_on_error,
            raise_on_exception=raise_on_exception,
            queue_size=queue_size,
        )
        record_bulk_results(results, receipt)
    receipt[INDEXED] = receipt.get(ATTEMPTED, 0) - len(receipt.get(REJECTED, []))
    return receipt


@mgp_read_proc
def connect(
    elastic_url: str,
    ca_certs: str = "",
    elastic_user: str = "",
    elastic_password: str = "",
) -> mgp_Record:
    (
        "Establishes connection with the Elasticsearch. This configuration needs to be specific to th"  # Continue literal.
        "e Elasticsearch deployment. Uses basic authentication\n    Args:\n        elastic_url (str): U"  # Continue literal.
        "RL for connecting to the Elasticsearch instance.\n        ca_certs (str): Path to the certifi"  # Continue literal.
        "cate file.\n        elastic_user (str): The user trying to connect to the Elasticsearch.\n    "  # Continue literal.
        "    elastic_password (str): User's password for connecting to the Elasticsearch.\n    Returns"  # Continue literal.
        ":\n        mgp.Record(connection_status=mgp.Map): Connection info.\n"
    )
    candidate = elasticsearch_Elasticsearch(
        hosts=elastic_url,
        ca_certs=ca_certs,
        basic_auth=(elastic_user, elastic_password),
    )
    published = False
    try:
        connection_status = dict(candidate.info())
        with connection_lock:
            displaced = list(connection.values())
            connection[CLIENT] = candidate
        published = True
    finally:
        # A candidate that failed validation is never published and its connection pool is released here.
        if not published:
            candidate.close()
    for displaced_client in displaced:
        displaced_client.close()
    logger.info(f"Client info: {connection_status}")
    computed_return_value = mgp_Record(connection_status=connection_status)
    return computed_return_value


def schema_section(schema: dict, path: tuple) -> dict:
    """Returns the nested mapping at path inside a loaded index schema, failing when the schema does not contain it."""
    section = schema
    for step in path:
        if isinstance(step, int):
            if not isinstance(section, list) or not 0 <= step < len(section):
                raise ValueError(f"Index schema has no section {path}.")
            section = section[step]
        else:
            if not isinstance(section, dict) or step not in section:
                raise ValueError(f"Index schema has no section {path}.")
            section = section.get(step, {})
    if not isinstance(section, dict):
        raise ValueError(f"Index schema section {path} is not a mapping.")
    return section


@mgp_read_proc
def create_index(
    context: mgp_ProcCtx,
    index_name: str,
    schema_path: str,
    schema_parameters: mgp_Map,
) -> mgp_Record:
    """Creates index with the given index name.
    Args:
        index_name (str): Name of the index that needs to be created.
        schema_path (str): Path to the schema from where it will be loaded.
        schema_parameters: Dict[str, Any]
            number_of_shards (int): Number of shards index will use.
            number_of_replicas (int): Number of replicas index will use.
            analyzer (str): Custom analyzer, can be set to any legal Elasticsearch analyzer.
    Returns:
       mgp.Map: Response message from Elasticsearch service.
    """
    # Read schema from the path given
    with open(schema_path, "r") as schema_file:
        schema_json = json_loads(schema_file.read())
    # Update default schema if specified
    if NUMBER_OF_SHARDS in schema_parameters:
        number_of_shards = schema_parameters.get(NUMBER_OF_SHARDS, 0)
        schema_section(schema_json, (SETTINGS, INDEX))[NUMBER_OF_SHARDS] = number_of_shards
        logger.info(f"Number of shards updated to: {number_of_shards}")
    if NUMBER_OF_REPLICAS in schema_parameters:
        number_of_replicas = schema_parameters.get(NUMBER_OF_REPLICAS, 0)
        schema_section(schema_json, (SETTINGS, INDEX))[NUMBER_OF_REPLICAS] = number_of_replicas
        logger.info(f"Number of replicas updated to: {number_of_replicas}")
    if ANALYZER in schema_parameters and INDEX_TYPE in schema_parameters:
        analyzer = schema_parameters.get(ANALYZER, "")
        schema_section(schema_json, (MAPPINGS, DYNAMIC_TEMPLATES, 1, STRING, MAPPING))[ANALYZER] = analyzer
        raw_template = MEM_CATEGORIES_HAS_RAW if schema_parameters.get(INDEX_TYPE, "") == VERTEX else MEM_TYPE_HAS_RAW
        schema_section(schema_json, (MAPPINGS, DYNAMIC_TEMPLATES, 0, raw_template, MAPPING))[ANALYZER] = analyzer
        logger.info(f"Analyzer set to: {analyzer}")
    logger.info(f"Schema dict: {schema_json}")
    computed_return_value = mgp_Record(
        response=dict(connected_client().indices.create(index=index_name, body=schema_json, ignore=400))
    )
    return computed_return_value


@mgp_read_proc
def index_db(
    context: mgp_ProcCtx,
    node_index: str,
    edge_index: str,
    thread_count: int = 1,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: float = 2.0,
    max_backoff: float = 600.0,
    yield_ok: bool = True,
    queue_size: int = 4,
) -> mgp_Record:
    # Now create iterable of documents that need to be indexed
    (
        "The method serializes all vertices and relationships that are in Memgraph DB to an ElasticSe"  # Continue literal.
        "arch schema.\n    Args:\n        context (mgp.ProcCtx): Reference to the executing context.\n  "  # Continue literal.
        "      node_index (str): The name of the node index. Can be used for both streaming and paral"  # Continue literal.
        "lel bulk.\n        edge_index (str): The name of the edge index. Can be used for both streami"  # Continue literal.
        "ng and parallel bulk.\n        chunk_size (int): The number of docs in one chunk sent to es ("  # Continue literal.
        "default: 500).\n        max_chunk_bytes (int): The maximum size of the request in bytes (defa"  # Continue literal.
        "ult: 100MB).\n        raise_on_error (bool): Raise bulkIndexError containing errors (as .erro"  # Continue literal.
        "rs) from the execution of the last chunk when some occur. By default we raise.\n        raise"  # Continue literal.
        "_on_exception (bool): If False then don’t propagate exceptions from call to bulk and just re"  # Continue literal.
        "port the items that failed as failed.\n        max_retries (int): Maximum number of times a d"  # Continue literal.
        "ocument will be retried when 429 is received, set to 0 (default) for no retries on 429.\n    "  # Continue literal.
        "    initial_backoff (float): The number of seconds we should wait before the first retry. An"  # Continue literal.
        "y subsequent retries will be powers of initial_backoff * 2**retry_number.\n        max_backof"  # Continue literal.
        "f (float): The maximum number of seconds a retry will wait.\n        yield_ok (float): If set"  # Continue literal.
        " to False will skip successful documents in the output.\n        thread_count (int): Size of "  # Continue literal.
        "the threadpool to use for the bulk requests.\n        queue_size (int): Size of the task queu"  # Continue literal.
        "e between the main thread (producing chunks to send) and the processing threads.\n    Returns"  # Continue literal.
        ":\n        mgp.Record(nodes=int, edges=int, rejected_nodes=mgp.List[mgp.Map], rejected_edges=m"  # Continue literal.
        "gp.List[mgp.Map]): Numbers of nodes and edges indexed, and every rejected item with its id, s"  # Continue literal.
        "tatus and error.\n"
    )
    # Settings are admitted before any graph object is read; documents are then serialized lazily as they are sent.
    if thread_count < 1:
        raise ValueError("Number of threads must be positive number. ")
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive number. ")
    if thread_count == 1:
        # Use streaming bulk
        node_receipt = elastic_search_streaming_bulk(
            database_vertex_documents(context),
            node_index,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            max_retries,
            initial_backoff,
            max_backoff,
            yield_ok,
        )
        edge_receipt = elastic_search_streaming_bulk(
            database_edge_documents(context),
            edge_index,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            max_retries,
            initial_backoff,
            max_backoff,
            yield_ok,
        )
    else:
        node_receipt = elastic_search_parallel_bulk(
            database_vertex_documents(context),
            node_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )
        edge_receipt = elastic_search_parallel_bulk(
            database_edge_documents(context),
            edge_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )

    # Counts are documents the server admitted; rejected items (non-raising settings) carry their own identity and reason.
    computed_return_value = mgp_Record(
        nodes=node_receipt.get(INDEXED, 0),
        edges=edge_receipt.get(INDEXED, 0),
        rejected_nodes=node_receipt.get(REJECTED, []),
        rejected_edges=edge_receipt.get(REJECTED, []),
    )
    return computed_return_value


@mgp_read_proc
def index(
    context: mgp_ProcCtx,
    createdObjects: mgp_List[mgp_Map],
    node_index: str,
    edge_index: str,
    thread_count: int = 1,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: float = 2.0,
    max_backoff: float = 600.0,
    yield_ok: bool = True,
    queue_size: int = 4,
) -> mgp_Record:
    # Now create iterable of documents that need to be indexed
    (
        "The method serializes all vertices and relationships that came into the Memgraph DB to an El"  # Continue literal.
        "asticSearch schema and sends streaming_bulk request to ElasticSearch's API.\n    Args:\n      "  # Continue literal.
        "  context (mgp.ProcCtx): Reference to the executing context.\n        createdObjects (List[Di"  # Continue literal.
        "ct[str, Any]]): List of all objects that were created and then sent as arguments to this met"  # Continue literal.
        'hod with the help of "create trigger".\n        node_index (str): The name of the node index.'  # Continue literal.
        "\n        edge_index (str): The name of the edge index.\n        chunk_size (int): The number "  # Continue literal.
        "of docs in one chunk sent to es (default: 500).\n        max_chunk_bytes (int): The maximum s"  # Continue literal.
        "ize of the request in bytes (default: 100MB).\n        raise_on_error (bool): Raise bulkIndex"  # Continue literal.
        "Error containing errors (as .errors) from the execution of the last chunk when some occur. B"  # Continue literal.
        "y default we raise.\n        raise_on_exception (bool): If False then don’t propagate excepti"  # Continue literal.
        "ons from call to bulk and just report the items that failed as failed.\n        max_retries ("  # Continue literal.
        "int): Maximum number of times a document will be retried when 429 is received, set to 0 (def"  # Continue literal.
        "ault) for no retries on 429.\n        initial_backoff (float): The number of seconds we shoul"  # Continue literal.
        "d wait before the first retry. Any subsequent retries will be powers of initial_backoff * 2*"  # Continue literal.
        "*retry_number.\n        max_backoff (float): The maximum number of seconds a retry will wait."  # Continue literal.
        "\n        yield_ok (float): If set to False will skip successful documents in the output.\n   "  # Continue literal.
        "     thread_count (int): Size of the threadpool to use for the bulk requests.\n        queue_"  # Continue literal.
        "size (int): Size of the task queue between the main thread (producing chunks to send) and th"  # Continue literal.
        "e processing threads.\n    Returns:\n        mgp.Record(nodes=int, edges=int, rejected_nodes=mgp.List[mg"  # Continue literal.
        "p.Map], rejected_edges=mgp.List[mgp.Map]): Numbers of nodes and edges indexed, and every rejec"  # Continue literal.
        "ted item with its id, status and error.\n"
    )
    # Settings are admitted before any graph object is read; documents are then serialized lazily as they are sent.
    if thread_count < 1:
        raise ValueError("Number of threads must be positive number. ")
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive number. ")
    if thread_count == 1:
        # Use streaming bulk
        node_receipt = elastic_search_streaming_bulk(
            triggered_vertex_documents(createdObjects),
            node_index,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            max_retries,
            initial_backoff,
            max_backoff,
            yield_ok,
        )
        edge_receipt = elastic_search_streaming_bulk(
            triggered_edge_documents(createdObjects),
            edge_index,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            max_retries,
            initial_backoff,
            max_backoff,
            yield_ok,
        )
    else:
        node_receipt = elastic_search_parallel_bulk(
            triggered_vertex_documents(createdObjects),
            node_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )
        edge_receipt = elastic_search_parallel_bulk(
            triggered_edge_documents(createdObjects),
            edge_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )

    # Counts are documents the server admitted; rejected items (non-raising settings) carry their own identity and reason.
    computed_return_value = mgp_Record(
        nodes=node_receipt.get(INDEXED, 0),
        edges=edge_receipt.get(INDEXED, 0),
        rejected_nodes=node_receipt.get(REJECTED, []),
        rejected_edges=edge_receipt.get(REJECTED, []),
    )
    return computed_return_value


@mgp_read_proc
def reindex(
    context: mgp_ProcCtx,
    source_index: mgp_Any,
    target_index: str,
    query: str,
    chunk_size: int = 500,
    scroll: str = "5m",
    op_type: mgp_Nullable[str] = None,
) -> mgp_Record:
    (
        "Reindex all documents that satisfy a given query from one index to another, potentially (if "  # Continue literal.
        "target_client is specified) on a different cluster. If you don’t specify the query you will "  # Continue literal.
        "reindex all the documents.\n    Args:\n        context (mgp.ProcCtx): Reference to the executi"  # Continue literal.
        "ng context.\n        updatatedObjects (List[Dict[str, Any]]): List of all objects that were u"  # Continue literal.
        'pdated and then sent as arguments to this method with the help of the "update trigger".\n    '  # Continue literal.
        "    source_index (Union[str, List[str]]): Identifies source index(or more of them) from wher"  # Continue literal.
        "e documents need to be indexed.\n        target_index (str): Identifies target index to where"  # Continue literal.
        " documents need to be indexed.\n        query (str): Query written as JSON.\n        chunk_siz"  # Continue literal.
        "e (int): Number of docs in one chunk sent to es (default: 500).\n        scroll (str): Specif"  # Continue literal.
        "ies how long a consistent view of the index should be maintained for scrolled search.\n      "  # Continue literal.
        "  op_type (Optional[str]): Explicit operation type. Defaults to ‘_index’. Data streams must "  # Continue literal.
        "be set to ‘create’. If not specified, will auto-detect if target_index is a data stream.\n   "  # Continue literal.
        " Returns:\n        response (str): Number of documents matched by a query in the source_index"  # Continue literal.
        ".\n"
    )
    # Null op_type is the SDK's own absence (it then auto-detects data streams); a supplied one must be a reindex operation.
    if op_type is not None and op_type not in REINDEX_OP_TYPES:
        raise ValueError(f"Unsupported reindex op_type {op_type!r}; expected one of {REINDEX_OP_TYPES}.")
    response = elasticsearch_helpers.reindex(
        client=connected_client(),
        source_index=source_index,
        target_index=target_index,
        query=json_loads(query),
        chunk_size=chunk_size,
        scroll=scroll,
        op_type=op_type,
    )
    computed_return_value = mgp_Record(response=str(response[0]))
    return computed_return_value


@mgp_read_proc
def scan(
    context: mgp_ProcCtx,
    index_name: str,
    query: str,
    scroll: str = "5m",
    raise_on_error: bool = True,
    preserve_order: bool = False,
    size: int = 1000,
    from_: int = 0,
    request_timeout: mgp_Nullable[float] = None,
    clear_scroll: bool = True,
    max_items: mgp_Nullable[int] = None,
) -> mgp_Record:
    (
        "Runs a query on a index specified by the index_name.\n    Args:\n        context (mgp.ProcCtx)"  # Continue literal.
        ": Reference to the executing context.\n        index_name (str): A name of the index.\n       "  # Continue literal.
        " query (str): Query written as JSON.\n        scroll (int): Specifies how long a consistent v"  # Continue literal.
        "iew of the index should be maintained for scrolled search.\n        raise_on_error (bool): Ra"  # Continue literal.
        "ises an exception (ScanError) if an error is encountered (some shards fail to execute). By d"  # Continue literal.
        "efault we raise.\n        preserve_order (bool): Don’t set the search_type to scan - this wil"  # Continue literal.
        "l cause the scroll to paginate with preserving the order. Note that this can be an extremely"  # Continue literal.
        " expensive operation and can easily lead to unpredictable results, use with caution.\n       "  # Continue literal.
        " size (int): Size (per shard) of the batch send at each iteration.\n        from (int): Start"  # Continue literal.
        "ing document offset. By default, you cannot page through more than 10,000 hits using the fro"  # Continue literal.
        "m and size parameters. To page through more hits, use the search_after parameter.\n        re"  # Continue literal.
        "quest_timeout (mgp.Nullable[float]): Explicit timeout for each call to scan.\n        clear_s"  # Continue literal.
        "croll (bool): Explicitly calls delete on the scroll id via the clear scroll API at the end o"  # Continue literal.
        "f the method on completion or error, defaults to true.\n        max_items (mgp.Nullable[int]): "  # Continue literal.
        "Most hits to return; null returns every hit. Reaching it stops the scroll and clears it whe"  # Continue literal.
        "n clear_scroll is true.\n    Returns:\n         mgp.Record(items=mgp.List[mgp.Map], complete=b"  # Continue literal.
        "ool): Hits matched by the query, and whether they are all of them (false when max_items stop"  # Continue literal.
        "ped the scan before the last hit).\n"
    )
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must not be negative.")
    hits = elasticsearch_helpers.scan(
        connected_client(),
        query=json_loads(query),
        index=index_name,
        scroll=scroll,
        raise_on_error=raise_on_error,
        preserve_order=preserve_order,
        size=size,
        request_timeout=request_timeout,
        clear_scroll=clear_scroll,
        from_=from_,
    )
    items = []
    complete = True
    try:
        for hit in hits:
            if max_items is not None and len(items) == max_items:
                # A hit beyond the caller's allowance exists, so the returned items are not the complete result.
                complete = False
                break
            items.append(hit)
    finally:
        # Closing the scan generator runs its clear_scroll cleanup when iteration stops early or fails.
        hits.close()

    computed_return_value = mgp_Record(items=items, complete=complete)
    return computed_return_value


@mgp_read_proc
def search(
    context: mgp_ProcCtx,
    index_name: str,
    query: str,
    size: int = 1000,
    from_: int = 0,
    aggregations: mgp_Nullable[mgp_Map] = None,
    aggs: mgp_Nullable[mgp_Map] = None,
) -> mgp_Record:
    """Searches for all documents by specifying query and index.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        index_name (str): A name of the index.
        query (str): Query written as JSON.
        aggregations (Optional[Mapping[str, Mapping[str, Any]]]): -
        aggs (Optional[Mapping[str, Mapping[str, Any]]]): -
    Returns:
         mgp.Record(items=mgp.List[mgp.Map]): List of all items matched by the specific query.
    """
    # Null aggregation maps are the SDK's absence and are omitted from the request body; the two names are aliases.
    if aggregations is not None and aggs is not None:
        raise ValueError("Pass aggregations or aggs, not both; they are the same request field.")
    response = connected_client().search(
        index=index_name,
        query=json_loads(query),
        aggregations=aggregations,
        aggs=aggs,
        size=size,
        from_=from_,
    )
    body = response.body
    found = body.get(HITS, {})
    result = {
        HITS: {HITS: found.get(HITS, []), TOTAL: found.get(TOTAL, {})},
        AGGREGATIONS: body.get(AGGREGATIONS, {}),
    }

    computed_return_value = mgp_Record(result=result)
    return computed_return_value
