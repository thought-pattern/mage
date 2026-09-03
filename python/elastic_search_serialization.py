"""Utilities for elastic search serialization."""

from datetime import datetime
from json import loads as json_loads

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

# Singleton client object
client: elasticsearch_Elasticsearch


# Helper method
def serialize_vertex(vertex: mgp_Vertex) -> dict[str, object]:
    """Serializes vertex to specified ElasticSearch schema.
    Args:
        vertex (mgp.Vertex): Reference to the vertex in Memgraph DB
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    doc = serialize_properties(vertex.properties.items())
    doc[MEM_CATEGORIES] = [label.name for label in vertex.labels]
    doc[INDEX] = {ID: f"{vertex.id}"}
    return doc


def serialize_edge(edge: mgp_Edge) -> dict[str, object]:
    """Serializes edge to specified ElasticSearch schema.
    Args:
        edge (mgp.Edge): Reference to the edge in Memgraph DB.
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    doc = serialize_properties(edge.properties.items())
    doc[MEM_TYPE] = edge.type.name
    doc[INDEX] = {ID: f"{edge.from_vertex.id}-{edge.id}"}
    return doc


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
            # Convert datetime to str, replace microsecond and add Z suffix(Zulu or zero offset) manually because Python doesn't
            # support it out of the box
            prop_value = f"{prop_value.replace(microsecond=0).isoformat()}Z"
            source[f"{prop_key}{MEM_DATE}"] = prop_value
        elif type(prop_value) in meme_mapping:
            source[f"{prop_key}{meme_mapping.get(type(prop_value), False)}"] = prop_value
    return source


def generate_document(context_object: mgp_Any) -> tuple[dict[str, object], str]:
    if context_object.get(EVENT_TYPE, False) == CREATED_VERTEX:
        computed_return_value = serialize_vertex(context_object.get(VERTEX, False)), VERTEX
        return computed_return_value
    elif context_object.get(EVENT_TYPE, False) == CREATED_EDGE:
        computed_return_value = serialize_edge(context_object.get(EDGE, False)), EDGE
        return computed_return_value
    raise ValueError(f"Unsupported trigger event: {context_object.get(EVENT_TYPE, False)}")


def generate_documents_from_triggered_objects(
    context_objects: list[mgp_Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    (
        "Generates vertices and edges documents for indexing and returns them as lists.\n    Args:\n   "  # Continue literal.
        "     context_objects (List[Dict[str, Any]]): Objects that are sent as parameters because of "  # Continue literal.
        "some trigger that was called. Trigger can be for update or for create.\n    Returns:\n        "  # Continue literal.
        "Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Serialized vertices and edges.\n"
    )
    vertices, edges = [], []
    for context_object in context_objects:
        if context_object.get(EVENT_TYPE, False) == CREATED_VERTEX:
            vertices.append(serialize_vertex(context_object.get(VERTEX, False)))
        elif context_object.get(EVENT_TYPE, False) == CREATED_EDGE:
            edges.append(serialize_edge(context_object.get(EDGE, False)))
    return vertices, edges


def generate_documents_from_db(
    context: mgp_ProcCtx,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Generates vertices and edges from the database.
    Args:
        context (mgp.ProcCtx): A reference to the context execution.
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Serialized vertices and edges.
    """
    vertices, edges = [], []
    for vertex in context.graph.vertices:
        vertices.append(serialize_vertex(vertex))
        for edge in vertex.out_edges:
            edges.append(serialize_edge(edge))

    return vertices, edges


def elastic_search_streaming_bulk(
    objects: list[mgp_Any],
    index: str,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: float = 2.0,
    max_backoff: float = 600.0,
    yield_ok: bool = True,
) -> bool:
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
        "lse will skip successful documents in the output.\n"
    )
    for _, _ in streaming_bulk(
        client=client,
        index=index,
        actions=objects,
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        yield_ok=yield_ok,
        raise_on_error=raise_on_error,
        raise_on_exception=raise_on_exception,
        max_retries=max_retries,
    ):
        pass
    return False


def elastic_search_parallel_bulk(
    objects: list[mgp_Any],
    index: str,
    thread_count: int = 8,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    queue_size: int = 4,
) -> bool:
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
        " (producing chunks to send) and the processing threads.\n"
    )
    for _, _ in parallel_bulk(
        client=client,
        index=index,
        actions=objects,
        thread_count=thread_count,
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        raise_on_error=raise_on_error,
        raise_on_exception=raise_on_exception,
        queue_size=queue_size,
    ):
        pass
    return False


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
    global client
    client = elasticsearch_Elasticsearch(
        hosts=elastic_url,
        ca_certs=ca_certs,
        basic_auth=(elastic_user, elastic_password),
    )
    logger.info(f"Client info: {client.info()}")
    computed_return_value = mgp_Record(connection_status=dict(client.info()))
    return computed_return_value


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
    global client
    # Read schema from the path given
    with open(schema_path, "r") as schema_file:
        schema_json = json_loads(schema_file.read())
    # Update default schema if specified
    if NUMBER_OF_SHARDS in schema_parameters:
        schema_json[SETTINGS][INDEX][NUMBER_OF_SHARDS] = schema_parameters[NUMBER_OF_SHARDS]
        logger.info(f"Number of shards updated to: {schema_parameters[NUMBER_OF_SHARDS]}")
    if NUMBER_OF_REPLICAS in schema_parameters:
        schema_json[SETTINGS][INDEX][NUMBER_OF_REPLICAS] = schema_parameters[NUMBER_OF_REPLICAS]
        logger.info(f"Number of replicas updated to: {schema_parameters[NUMBER_OF_REPLICAS]}")
    if ANALYZER in schema_parameters and INDEX_TYPE in schema_parameters:
        schema_json[MAPPINGS][DYNAMIC_TEMPLATES][1][STRING][MAPPING][ANALYZER] = schema_parameters[ANALYZER]
        if schema_parameters[INDEX_TYPE] == VERTEX:
            schema_json[MAPPINGS][DYNAMIC_TEMPLATES][0][MEM_CATEGORIES_HAS_RAW][MAPPING][ANALYZER] = schema_parameters[ANALYZER]
        else:
            schema_json[MAPPINGS][DYNAMIC_TEMPLATES][0][MEM_TYPE_HAS_RAW][MAPPING][ANALYZER] = schema_parameters[ANALYZER]
        logger.info(f"Analyzer set to: {schema_parameters[ANALYZER]}")
    logger.info(f"Schema dict: {schema_json}")
    computed_return_value = mgp_Record(response=dict(client.indices.create(index=index_name, body=schema_json, ignore=400)))
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
        ":\n        mgp.Record(): Returns number of nodes and edges.\n"
    )  # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents_from_db(context)

    if thread_count < 1:
        raise ValueError("Number of threads must be positive number. ")
    elif thread_count == 1:
        # Use streaming bulk
        # Send nodes on indexing
        elastic_search_streaming_bulk(
            nodes,
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
        # Send edges on indexing
        elastic_search_streaming_bulk(
            edges,
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
        # Send nodes on indexing
        elastic_search_parallel_bulk(
            nodes,
            node_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )
        elastic_search_parallel_bulk(
            edges,
            edge_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )

    computed_return_value = mgp_Record(nodes=len(nodes), edges=len(edges))
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
        "e processing threads.\n    Returns:\n        mgp.Record(): Returns number of nodes and edges.\n"
    )  # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents_from_triggered_objects(createdObjects)

    if thread_count < 1:
        raise ValueError("Number of threads must be positive number. ")
    elif thread_count == 1:
        # Send nodes on indexing
        elastic_search_streaming_bulk(
            nodes,
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
        # Send edges on indexing
        elastic_search_streaming_bulk(
            edges,
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
        # Send nodes on indexing
        elastic_search_parallel_bulk(
            nodes,
            node_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )
        elastic_search_parallel_bulk(
            edges,
            edge_index,
            thread_count,
            chunk_size,
            max_chunk_bytes,
            raise_on_error,
            raise_on_exception,
            queue_size,
        )
    computed_return_value = mgp_Record(nodes=len(nodes), edges=len(edges))
    return computed_return_value


@mgp_read_proc
def reindex(
    context: mgp_ProcCtx,
    source_index: mgp_Any,
    target_index: str,
    query: str,
    chunk_size: int = 500,
    scroll: str = "5m",
    op_type: mgp_Nullable[str] = False,
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
    global client
    response = elasticsearch_helpers.reindex(
        client=client,
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
    request_timeout: mgp_Nullable[float] = False,
    clear_scroll: bool = False,
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
        "f the method on completion or error, defaults to true.\n    Returns:\n         mgp.Record(item"  # Continue literal.
        "s=mgp.List[mgp.Map]): List of all items matched by the specific query.\n"
    )
    global client
    response = elasticsearch_helpers.scan(
        client,
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
    for item in response:
        if INTERNAL_SOURCE in item and INDEX in item.get(INTERNAL_SOURCE, {}):
            item[ID] = item.get(INTERNAL_SOURCE, {})[INDEX][ID]
            item.get(INTERNAL_SOURCE, {}).pop(INDEX, False)
        items.append(item)

    computed_return_value = mgp_Record(items=items)
    return computed_return_value


@mgp_read_proc
def search(
    context: mgp_ProcCtx,
    index_name: str,
    query: str,
    size: int = 1000,
    from_: int = 0,
    aggregations: mgp_Nullable[mgp_Map] = False,
    aggs: mgp_Nullable[mgp_Map] = False,
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
    global client
    response = client.search(
        index=index_name,
        query=json_loads(query),
        aggregations=aggregations,
        aggs=aggs,
        size=size,
        from_=from_,
    )
    hits = []
    for hit in response[HITS][HITS]:
        if INTERNAL_SOURCE in hit and INDEX in hit[INTERNAL_SOURCE]:
            hit[ID] = hit[INTERNAL_SOURCE][INDEX][ID]
            hit[INTERNAL_SOURCE].pop(INDEX, False)
        hits.append(hit)

    result = {}
    result[HITS] = {HITS: hits, TOTAL: response[HITS][TOTAL]}
    if AGGREGATIONS in response:
        result[AGGREGATIONS] = response[AGGREGATIONS]
    else:
        result[AGGREGATIONS] = dict()

    computed_return_value = mgp_Record(result=result)
    return computed_return_value
