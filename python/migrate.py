"""Utilities for migrate."""

from base64 import b64encode as base64_b64encode
from csv import reader as csv_reader_2
from datetime import date as datetime_date
from datetime import datetime as datetime_datetime
from datetime import time as datetime_time
from datetime import timedelta as datetime_timedelta
from decimal import Decimal
from functools import partial
from hashlib import sha256 as hashlib_sha256
from io import BytesIO as io_BytesIO
from io import TextIOWrapper as io_TextIOWrapper
from ipaddress import ip_address
from itertools import islice
from json import load as json_load
from os import getenv as os_getenv
from re import compile as re_compile
from threading import Lock

from boto3 import client as boto3_client
from duckdb import connect as duckDB_connect
from gqlalchemy import Memgraph
from mgp import Any as mgp_Any
from mgp import Map as mgp_Map
from mgp import Nullable as mgp_Nullable
from mgp import Record as mgp_Record
from mgp import add_batch_read_proc as mgp_add_batch_read_proc
from mysql import connector as mysql_connector
from neo4j import GraphDatabase
from neo4j.time import Date as Neo4jDate
from neo4j.time import DateTime as Neo4jDateTime
from oracledb import connect as oracledb_connect
from psycopg2 import connect as psycopg2_connect
from pyarrow import flight
from pyodbc import connect as pyodbc_connect
from requests import get as requests_get


class Constants:
    BATCH_SIZE = 1000
    DATABASE = "database"
    HOST = "host"
    I_COLUMN_NAME = 0
    PASSWORD = "password"
    PORT = "port"
    RESULT = "result"
    URI_SCHEME = "uri_scheme"
    USERNAME = "username"
    # Memgraph integers are signed 64-bit.
    INT64_MIN = -(2**63)
    INT64_MAX = 2**63 - 1
    # Arrow Flight transport: verified TLS unless `tls: false` names a loopback host.
    TLS = "tls"
    TLS_ROOT_CERTS = "tls_root_certs"
    # ServiceNow connect and read deadline per request; the default matches
    # ServiceNow's 60-second REST transaction quota, after which the instance
    # itself cancels the request. The procedure config may override it.
    TIMEOUT = "timeout"
    SERVICENOW_TIMEOUT_SECONDS = 60
    SERVICENOW_PAGE_LIMIT = "sysparm_limit"


# A table reference: one identifier, optionally schema-qualified, where each
# part is a bare word or a backtick, double-quote or bracket quoted name.
SQL_IDENTIFIER = r'(?:[^\W\d][\w$#@]*|`(?:[^`]|``)+`|"(?:[^"]|"")+"|\[(?:[^\]]|\]\])+\])'
SQL_TABLE_REFERENCE = re_compile(rf"\s*{SQL_IDENTIFIER}(?:\s*\.\s*{SQL_IDENTIFIER})*\s*")
# Cypher shorthand selectors `(:Label)` and `[:REL_TYPE]`, bare or backtick quoted.
CYPHER_NAME = r"(\w+|`(?:[^`]|``)+`)"
CYPHER_NODE_SHORTHAND = re_compile(rf"\s*\(\s*:{CYPHER_NAME}\s*\)\s*")
CYPHER_RELATIONSHIP_SHORTHAND = re_compile(rf"\s*\[\s*:{CYPHER_NAME}\s*\]\s*")


class MigrationStreams:
    """Open migration streams of one batched procedure, keyed by its exact arguments.

    Memgraph calls the initializer once per stream and then the batch function
    with the same arguments until it returns no rows, so the arguments are the
    only stream identity a batch call can recover. A second stream with
    identical arguments could not be told apart from the first and is refused
    at admission; reservation is atomic, so two concurrent identical calls
    cannot both acquire resources.

    A stream releases everything it owns when it is exhausted (committing first),
    when a fetch or conversion fails, and when its acquisition fails. The
    registered cleanup callback receives no arguments, and released Memgraph
    versions also call it before every initializer and on every cursor reset,
    so it cannot tell which stream ended and releases nothing.
    """

    def __init__(self, procedure: str):
        self.procedure = procedure
        self.lock = Lock()
        self.streams = {}

    def open(self, key: str, acquire, *arguments) -> None:
        """Admit one stream under key; acquire fills it, and a failed acquisition releases what it acquired."""
        # closers run in reverse acquisition order; commit only after exhaustion
        stream = {"closers": [], "commit": False, "fetch": False}
        with self.lock:
            if key in self.streams:
                raise RuntimeError(
                    "Migrate module with these parameters is already running. "
                    "Please wait for it to finish before starting a new one."
                )
            self.streams[key] = stream
        admitted = False
        try:
            acquire(stream, *arguments)
            admitted = True
        finally:
            if not admitted:
                self.release(key, completed=False)

    def next_batch(self, key: str) -> list[mgp_Record]:
        """Fetch the next batch of the stream opened under key, releasing it at exhaustion or on failure."""
        with self.lock:
            stream = self.streams.get(key, {})
        fetch = stream.get("fetch", False)
        if not fetch:
            raise RuntimeError(f"migrate.{self.procedure} has no open stream for these arguments")
        failed = True
        try:
            records = fetch()
            failed = False
        finally:
            if failed:
                self.release(key, completed=False)
        if not records:
            self.release(key, completed=True)
        return records

    def release(self, key: str, completed: bool) -> None:
        with self.lock:
            stream = self.streams.pop(key, {})
        steps = list(reversed(stream.get("closers", [])))
        commit = stream.get("commit", False)
        if completed and commit:
            steps.insert(0, commit)
        run_in_order(steps)


def run_in_order(steps: list) -> None:
    """Run every step in order even when one raises; a failure propagates after the remaining steps ran."""
    if not steps:
        return
    try:
        steps[0]()
    finally:
        run_in_order(steps[1:])


def stream_key(*arguments) -> str:
    """Stream identity: the exact procedure arguments, which Memgraph passes unchanged to every batch call."""
    computed_return_value = hashlib_sha256(repr(arguments).encode("utf-8")).hexdigest()
    return computed_return_value


def fetch_sql_records(cursor, column_names: list, name_cells) -> list[mgp_Record]:
    records = [mgp_Record(row=name_cells(row, column_names)) for row in cursor.fetchmany(Constants.BATCH_SIZE)]
    return records


def fetch_iterator_records(rows, convert) -> list[mgp_Record]:
    records = [mgp_Record(row=convert(row)) for row in islice(rows, Constants.BATCH_SIZE)]
    return records


# MYSQL

mysql_streams = MigrationStreams("mysql")


def open_mysql_stream(stream: dict, table_or_sql: str, config: mgp_Map, config_path: str, params) -> None:
    if params:
        check_params_type(params)
    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    connection = mysql_connector.connect(**effective_config(config, config_path))
    # close() also ends a connection that still has unread rows; commit only runs after exhaustion
    stream["closers"].append(connection.close)
    stream["commit"] = connection.commit
    cursor = connection.cursor()
    if params:
        cursor.execute(table_or_sql, params=params)
    else:
        cursor.execute(table_or_sql)

    column_names = [column[Constants.I_COLUMN_NAME] for column in cursor.description]
    stream["fetch"] = partial(fetch_sql_records, cursor, column_names, name_row_cells_mysql)


def init_migrate_mysql(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    key = stream_key(table_or_sql, config, config_path, params)
    mysql_streams.open(key, open_mysql_stream, table_or_sql, config, config_path, params)


def mysql(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    With migrate.mysql you can access MySQL and execute queries.
    The result table is converted into a stream, and returned rows can be
    used to create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will
    overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters
                   (as in mysql.connector.connect)
    :param config_path: Path to the JSON file containing configuration
                        parameters (as in mysql.connector.connect)
    :param params: Optionally, queries may be parameterized. In that case,
                   `params` provides parameter values
    :return: The result table as a stream of rows
    """
    records = mysql_streams.next_batch(stream_key(table_or_sql, config, config_path, params))
    return records


def cleanup_migrate_mysql():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(mysql, init_migrate_mysql, cleanup_migrate_mysql)

# SQL SERVER

sql_server_streams = MigrationStreams("sql_server")


def open_sql_server_stream(stream: dict, table_or_sql: str, config: mgp_Map, config_path: str, params) -> None:
    if params:
        check_params_type(params, (list, tuple))
    else:
        params = []
    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    connection = pyodbc_connect(**effective_config(config, config_path))
    stream["closers"].append(connection.close)
    stream["commit"] = connection.commit
    cursor = connection.cursor()
    stream["closers"].append(cursor.close)
    cursor.execute(table_or_sql, *params)

    column_names = [column[Constants.I_COLUMN_NAME] for column in cursor.description]
    stream["fetch"] = partial(fetch_sql_records, cursor, column_names, name_row_cells)


def init_migrate_sql_server(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    key = stream_key(table_or_sql, config, config_path, params)
    sql_server_streams.open(key, open_sql_server_stream, table_or_sql, config, config_path, params)


def sql_server(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    With migrate.sql_server you can access SQL Server and execute queries.
    The result table is converted into a stream, and returned rows can be
    used to create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will
    overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in pyodbc.connect)
    :param config_path: Path to the JSON file containing configuration
                        parameters (as in pyodbc.connect)
    :param params: Optionally, queries may be parameterized. In that case,
                   `params` provides parameter values
    :return: The result table as a stream of rows
    """
    records = sql_server_streams.next_batch(stream_key(table_or_sql, config, config_path, params))
    return records


def cleanup_migrate_sql_server():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(sql_server, init_migrate_sql_server, cleanup_migrate_sql_server)

# Oracle DB

oracle_db_streams = MigrationStreams("oracle_db")


def open_oracle_db_stream(stream: dict, table_or_sql: str, config: mgp_Map, config_path: str, params) -> None:
    if params:
        check_params_type(params)
    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql}"

    oracle_config = effective_config(config, config_path)
    # To prevent query execution from hanging
    oracle_config["disable_oob"] = True

    connection = oracledb_connect(**oracle_config)
    stream["closers"].append(connection.close)
    stream["commit"] = connection.commit
    cursor = connection.cursor()
    stream["closers"].append(cursor.close)

    if not params:
        cursor.execute(table_or_sql)
    elif isinstance(params, (list, tuple)):
        cursor.execute(table_or_sql, params)
    else:
        cursor.execute(table_or_sql, **params)

    column_names = [column[Constants.I_COLUMN_NAME] for column in cursor.description]
    stream["fetch"] = partial(fetch_sql_records, cursor, column_names, name_row_cells)


def init_migrate_oracle_db(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    key = stream_key(table_or_sql, config, config_path, params)
    oracle_db_streams.open(key, open_oracle_db_stream, table_or_sql, config, config_path, params)


def oracle_db(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    With migrate.oracle_db you can access Oracle DB and execute queries.
    The result table is converted into a stream, and returned rows can be
    used to create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will
    overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in oracledb.connect)
    :param config_path: Path to the JSON file containing configuration
                        parameters (as in oracledb.connect)
    :param params: Optionally, queries may be parameterized. In that case,
                   `params` provides parameter values
    :return: The result table as a stream of rows
    """
    records = oracle_db_streams.next_batch(stream_key(table_or_sql, config, config_path, params))
    return records


def cleanup_migrate_oracle_db():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(oracle_db, init_migrate_oracle_db, cleanup_migrate_oracle_db)


# PostgreSQL

postgres_streams = MigrationStreams("postgresql")


def open_postgresql_stream(stream: dict, table_or_sql: str, config: mgp_Map, config_path: str, params) -> None:
    if params:
        check_params_type(params, (list, tuple))
    else:
        params = []
    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    connection = psycopg2_connect(**effective_config(config, config_path))
    stream["closers"].append(connection.close)
    stream["commit"] = connection.commit
    cursor = connection.cursor()
    stream["closers"].append(cursor.close)
    cursor.execute(table_or_sql, params)

    column_names = [column.name for column in cursor.description]
    stream["fetch"] = partial(fetch_sql_records, cursor, column_names, name_row_cells)


def init_migrate_postgresql(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    key = stream_key(table_or_sql, config, config_path, params)
    postgres_streams.open(key, open_postgresql_stream, table_or_sql, config, config_path, params)


def postgresql(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    With migrate.postgresql you can access PostgreSQL and execute queries.
    The result table is converted into a stream, and returned rows can be
    used to create graph structures. Config must be at least empty map.
    If config_path is passed, every key,value pair from JSON file will
    overwrite any values in config file.

    :param table_or_sql: Table name or an SQL query
    :param config: Connection configuration parameters (as in psycopg2.connect)
    :param config_path: Path to the JSON file containing configuration
                        parameters (as in psycopg2.connect)
    :param params: Optionally, queries may be parameterized. In that case,
                   `params` provides parameter values
    :return: The result table as a stream of rows
    """
    records = postgres_streams.next_batch(stream_key(table_or_sql, config, config_path, params))
    return records


def cleanup_migrate_postgresql():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(postgresql, init_migrate_postgresql, cleanup_migrate_postgresql)


# S3

s3_streams = MigrationStreams("s3")


def open_s3_stream(stream: dict, file_path: str, config: mgp_Map, config_path: str) -> None:
    s3_config = effective_config(config, config_path)

    # Extract S3 bucket and key
    if not file_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/path'.")

    file_path_no_protocol = file_path[5:]
    bucket_name, *key_parts = file_path_no_protocol.split("/")
    s3_key = "/".join(key_parts)

    # Pass only configured credentials; boto3 resolves absent ones through its own provider chain.
    client_options = {}
    for option, variable in (
        ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
        ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
        ("aws_session_token", "AWS_SESSION_TOKEN"),
        ("region_name", "AWS_REGION"),
    ):
        value = s3_config.get(option, os_getenv(variable, ""))
        if value:
            client_options[option] = value
    s3_client = boto3_client("s3", **client_options)
    stream["closers"].append(s3_client.close)

    # Fetch and read file as a streaming object
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    # Convert binary stream to text stream; closing it closes the HTTP body
    text_stream = io_TextIOWrapper(response.get("Body", io_BytesIO()), encoding="utf-8")
    stream["closers"].append(text_stream.close)

    csv_reader = csv_reader_2(text_stream)
    # First row contains column names; an empty object has none and no rows
    column_names = next(csv_reader, [])
    stream["fetch"] = partial(fetch_iterator_records, csv_reader, partial(name_row_cells, column_names=column_names))


def init_migrate_s3(
    file_path: str,
    config: mgp_Map,
    config_path: str = "",
):
    """
    Initialize an S3 connection and prepare to stream a CSV file.

    :param file_path: S3 file path in the format
                      's3://bucket-name/path/to/file.csv'
    :param config: Configuration map containing AWS credentials
                   (access_key, secret_key, region, etc.)
    :param config_path: Path to a JSON file containing configuration parameters
    """
    s3_streams.open(stream_key(file_path, config, config_path), open_s3_stream, file_path, config, config_path)


def s3(
    file_path: str,
    config: mgp_Map,
    config_path: str = "",
) -> list[mgp_Record]:
    """
    Fetch rows from an S3 CSV file in batches.

    :param file_path: S3 file path in the format
                      's3://bucket-name/path/to/file.csv'
    :param config: AWS S3 connection parameters (AWS credentials, region, etc.)
    :param config_path: Optional path to a JSON file containing AWS credentials
    :return: The result table as a stream of rows
    """
    records = s3_streams.next_batch(stream_key(file_path, config, config_path))
    return records


def cleanup_migrate_s3():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(s3, init_migrate_s3, cleanup_migrate_s3)


# Neo4j

neo4j_streams = MigrationStreams("neo4j")


def open_neo4j_stream(stream: dict, label_or_rel_or_query: str, config: mgp_Map, config_path: str, params) -> None:
    neo4j_config = effective_config(config, config_path)
    query = formulate_cypher_query(label_or_rel_or_query)

    username = neo4j_config.get(Constants.USERNAME, "neo4j")
    password = neo4j_config.get(Constants.PASSWORD, "password")
    database = neo4j_config.get(Constants.DATABASE, "")

    driver = GraphDatabase.driver(build_neo4j_uri(neo4j_config), auth=(username, password))
    stream["closers"].append(driver.close)

    # An absent database selects the server's default database
    if database:
        session = driver.session(database=database)
    else:
        session = driver.session()
    stream["closers"].append(session.close)

    # Neo4j expects the parameters as a map
    result = session.run(query, parameters=params if params else {})
    stream["fetch"] = partial(fetch_iterator_records, iter(result), convert_neo4j_record)


def init_migrate_neo4j(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    key = stream_key(label_or_rel_or_query, config, config_path, params)
    neo4j_streams.open(key, open_neo4j_stream, label_or_rel_or_query, config, config_path, params)


def neo4j(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    Migrate data from Neo4j to Memgraph. Can migrate a specific node label, relationship type, or execute a custom Cypher query.

    :param label_or_rel_or_query: Node label, relationship type, or a Cypher query
    :param config: Connection configuration for Neo4j
    :param config_path: Path to a JSON file containing connection parameters
    :param params: Optional query parameters
    :return: Stream of rows from Neo4j
    """
    records = neo4j_streams.next_batch(stream_key(label_or_rel_or_query, config, config_path, params))
    return records


def cleanup_migrate_neo4j():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(neo4j, init_migrate_neo4j, cleanup_migrate_neo4j)


# Arrow Flight

flight_streams = MigrationStreams("arrow_flight")


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        loopback = ip_address(host).is_loopback
    except ValueError:
        loopback = False
    return loopback


def connect_arrow_flight(config: dict):
    """Connect over verified TLS; plaintext only when `tls: false` explicitly names a loopback host.

    Every call carries the configured credentials in a Basic authorization
    header, so a remote endpoint must be reached over TLS. `tls_root_certs`
    optionally names a PEM file with the roots that verify the server.
    """
    host = config.get(Constants.HOST, "")
    port = config.get(Constants.PORT, "")
    if not host or not port:
        raise ValueError("Arrow Flight config requires host and port")

    if config.get(Constants.TLS, True):
        client_options = {}
        root_certs_path = config.get(Constants.TLS_ROOT_CERTS, "")
        if root_certs_path:
            with open(root_certs_path, "rb") as root_certs:
                client_options["tls_root_certs"] = root_certs.read()
        client = flight.connect(f"grpc+tls://{host}:{port}", **client_options)
        return client

    if not is_loopback_host(str(host)):
        raise ValueError(
            "Arrow Flight sends credentials with every call, so plaintext (tls: false) is only allowed to a loopback host"
        )
    client = flight.connect(f"grpc+tcp://{host}:{port}")
    return client


def open_arrow_flight_stream(stream: dict, query: str, config: mgp_Map, config_path: str) -> None:
    flight_config = effective_config(config, config_path)
    username = flight_config.get(Constants.USERNAME, "")
    password = flight_config.get(Constants.PASSWORD, "")

    # Encode credentials
    auth_string = f"{username}:{password}".encode("utf-8")
    encoded_auth = base64_b64encode(auth_string).decode("utf-8")

    client = connect_arrow_flight(flight_config)
    stream["closers"].append(client.close)

    # Authenticate
    options = flight.FlightCallOptions(headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))])

    flight_info = client.get_flight_info(flight.FlightDescriptor.for_command(query), options)
    stream["fetch"] = partial(fetch_iterator_records, fetch_flight_data(client, flight_info, options), convert_row_types)


def init_migrate_arrow_flight(
    query: str,
    config: mgp_Map,
    config_path: str = "",
):
    flight_streams.open(stream_key(query, config, config_path), open_arrow_flight_stream, query, config, config_path)


def fetch_flight_data(client, flight_info, options):
    """
    Efficiently fetches data in batches from Arrow Flight using RecordBatchReader.
    This prevents high memory usage by avoiding full table loading.
    """
    for endpoint in flight_info.endpoints:
        reader = client.do_get(endpoint.ticket, options)  # Stream the data
        for chunk in reader:  # Iterate over RecordBatches
            batch = chunk.data  # Convert each batch to an Arrow Table
            yield from batch.to_pylist()  # Convert to row dictionaries on demand


def arrow_flight(
    query: str,
    config: mgp_Map,
    config_path: str = "",
) -> list[mgp_Record]:
    """
    Execute a SQL query on Arrow Flight and stream results into Memgraph.

    :param query: SQL query to execute
    :param config: Arrow Flight connection configuration: host, port, username,
                   password, tls (default true; false is accepted only for a
                   loopback host) and tls_root_certs (optional PEM file path)
    :param config_path: Path to a JSON config file
    :return: Stream of rows from Arrow Flight
    """
    records = flight_streams.next_batch(stream_key(query, config, config_path))
    return records


def cleanup_migrate_arrow_flight():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(arrow_flight, init_migrate_arrow_flight, cleanup_migrate_arrow_flight)


# DuckDB

duckdb_streams = MigrationStreams("duckdb")


def open_duckdb_stream(stream: dict, query: str, setup_queries) -> None:
    # Ensure a fresh in-memory DuckDB instance for each query
    connection = duckDB_connect()
    stream["closers"].append(connection.close)
    cursor = connection.cursor()
    stream["closers"].append(cursor.close)
    for setup_query in setup_queries if setup_queries else []:
        cursor.execute(setup_query)

    cursor.execute(query)

    column_names = [desc[0] for desc in cursor.description]
    stream["fetch"] = partial(fetch_sql_records, cursor, column_names, name_row_cells)


def init_migrate_duckdb(query: str, setup_queries: mgp_Nullable[list[str]] = False):
    """
    Initialize an in-memory DuckDB connection and execute the query.

    :param query: SQL query to execute
    :param setup_queries: Optional list of setup queries to execute before the main query
    """
    duckdb_streams.open(stream_key(query, setup_queries), open_duckdb_stream, query, setup_queries)


def duckdb(query: str, setup_queries: mgp_Nullable[list[str]] = False) -> list[mgp_Record]:
    """
    Fetch rows from DuckDB in batches.

    :param query: SQL query to execute
    :param setup_queries: Optional list of setup queries to execute before the main query
    :return: The result table as a stream of rows
    """
    records = duckdb_streams.next_batch(stream_key(query, setup_queries))
    return records


def cleanup_migrate_duckdb():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(duckdb, init_migrate_duckdb, cleanup_migrate_duckdb)


# Memgraph

memgraph_streams = MigrationStreams("memgraph")


def mapping_row(row) -> dict:
    """Admit one source row; a stream row must be a map of column name to value."""
    if not isinstance(row, dict):
        raise TypeError(f"source row must be a map, received {type(row).__name__}")
    return row


def open_memgraph_stream(stream: dict, label_or_rel_or_query: str, config: mgp_Map, config_path: str, params) -> None:
    query = formulate_cypher_query(label_or_rel_or_query)
    memgraph_db = Memgraph(**effective_config(config, config_path))
    # The row generator holds the client connection; closing it and dropping the
    # stream releases that connection (gqlalchemy exposes no explicit close).
    rows = memgraph_db.execute_and_fetch(query, params if params else {})
    stream["closers"].append(rows.close)
    stream["fetch"] = partial(fetch_iterator_records, rows, mapping_row)


def init_migrate_memgraph(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    key = stream_key(label_or_rel_or_query, config, config_path, params)
    memgraph_streams.open(key, open_memgraph_stream, label_or_rel_or_query, config, config_path, params)


def memgraph(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    Migrate data from Memgraph to another Memgraph instance. Can migrate a specific node label,
    relationship type, or execute a custom Cypher query.

    :param label_or_rel_or_query: Node label, relationship type, or a Cypher query
    :param config: Connection configuration for Memgraph
    :param config_path: Path to a JSON file containing connection parameters
    :param params: Optional query parameters
    :return: Stream of rows from Memgraph
    """
    records = memgraph_streams.next_batch(stream_key(label_or_rel_or_query, config, config_path, params))
    return records


def cleanup_migrate_memgraph():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(memgraph, init_migrate_memgraph, cleanup_migrate_memgraph)


# ServiceNow

servicenow_streams = MigrationStreams("servicenow")


def fetch_servicenow_page(url: str, request: dict, params) -> tuple[list, str]:
    """One page of a ServiceNow response: its result rows and the URL of the next page ("" when last)."""
    response = requests_get(url, params=params, **request)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get(Constants.RESULT, []), list):
        raise ValueError("ServiceNow response must be a map with a result list")
    page = (payload.get(Constants.RESULT, []), response.links.get("next", {}).get("url", ""))
    return page


def servicenow_rows(first_page: tuple[list, str], request: dict):
    """Rows of every page, following the Link-header pagination contract one page at a time."""
    rows, next_url = first_page
    yield from rows
    while next_url:
        rows, next_url = fetch_servicenow_page(next_url, request, {})
        yield from rows


def open_servicenow_stream(stream: dict, endpoint: str, config: mgp_Map, config_path: str, params) -> None:
    servicenow_config = effective_config(config, config_path)
    request = {
        "auth": (servicenow_config.get(Constants.USERNAME, ""), servicenow_config.get(Constants.PASSWORD, "")),
        "headers": {"Accept": "application/json"},
        "timeout": servicenow_config.get(Constants.TIMEOUT, Constants.SERVICENOW_TIMEOUT_SECONDS),
    }
    # Pages hold at most one batch unless the caller chose its own page size.
    query = {Constants.SERVICENOW_PAGE_LIMIT: Constants.BATCH_SIZE}
    if params:
        check_params_type(params, (dict,))
        query.update(params)

    first_page = fetch_servicenow_page(endpoint, request, query)
    if not first_page[0]:
        raise ValueError("No data found in ServiceNow response")
    stream["fetch"] = partial(fetch_iterator_records, servicenow_rows(first_page, request), mapping_row)


def init_migrate_servicenow(
    endpoint: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    """
    Initialize the connection to the ServiceNow REST API and fetch the first page of JSON data.

    :param endpoint: ServiceNow API endpoint (full URL)
    :param config: Configuration map containing authentication details (username, password) and an optional
                   per-request timeout in seconds
    :param config_path: Optional path to a JSON file containing authentication details
    :param params: Optional query parameters for filtering results
    """
    key = stream_key(endpoint, config, config_path, params)
    servicenow_streams.open(key, open_servicenow_stream, endpoint, config, config_path, params)


def servicenow(
    endpoint: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    Fetch rows from the ServiceNow REST API in batches, following its pagination links.

    :param endpoint: ServiceNow API endpoint (full URL)
    :param config: Authentication details (username, password) and an optional per-request timeout in seconds
    :param config_path: Optional path to a JSON file containing authentication details
    :param params: Optional query parameters for filtering results
    :return: The result data as a stream of rows
    """
    records = servicenow_streams.next_batch(stream_key(endpoint, config, config_path, params))
    return records


def cleanup_migrate_servicenow():
    """Cleanup callback required by the batched-procedure protocol; see MigrationStreams."""


mgp_add_batch_read_proc(servicenow, init_migrate_servicenow, cleanup_migrate_servicenow)


def formulate_cypher_query(label_or_rel_or_query: str) -> str:
    # The anchored shorthand forms are recognised first, so whitespace inside
    # them is not mistaken for a multi-word query.
    node_match = CYPHER_NODE_SHORTHAND.fullmatch(label_or_rel_or_query)
    if node_match:
        label = node_match.group(1)
        computed_return_value = f"MATCH (n:{label}) RETURN labels(n) as labels, properties(n) as properties"
        return computed_return_value

    rel_match = CYPHER_RELATIONSHIP_SHORTHAND.fullmatch(label_or_rel_or_query)
    if rel_match:
        rel_type = rel_match.group(1)
        computed_return_value = f"""
    MATCH (n)-[r:{rel_type}]->(m)
    RETURN
        labels(n) as from_labels,
        labels(m) as to_labels,
        properties(n) as from_properties,
        properties(r) as edge_properties,
        properties(m) as to_properties
    """
        return computed_return_value
    return label_or_rel_or_query  # Assume it's a valid query


def query_is_table(table_or_sql: str) -> bool:
    """A table reference is a single, optionally qualified, bare or quoted identifier; anything else is SQL."""
    computed_return_value = bool(SQL_TABLE_REFERENCE.fullmatch(table_or_sql))
    return computed_return_value


def effective_config(config: mgp_Map, config_path: str) -> dict[str, object]:
    """The stream's configuration, read once at admission: the map overlaid by the JSON file at config_path."""
    combined = dict(config)
    if not config_path:
        return combined
    try:
        with open(config_path, "r") as file:
            file_config = json_load(file)
    except (OSError, ValueError) as err:
        raise OSError(f"Could not open/read config file {config_path!r}") from err
    if not isinstance(file_config, dict):
        raise ValueError(f"Config file {config_path!r} must contain a JSON object")
    combined.update(file_config)
    return combined


def convert_decimal(value: Decimal):
    """Admit a Decimal into Memgraph, which has no decimal type, without silently changing its value.

    The value becomes a float when that float's shortest decimal form equals the
    source value (e.g. DECIMAL 123.45), an integer when it is integral and fits
    Memgraph's signed 64-bit range (e.g. identifiers beyond float precision),
    and is refused otherwise.
    """
    approximation = float(value)
    if Decimal(repr(approximation)) == value:
        return approximation
    if value == value.to_integral_value() and Constants.INT64_MIN <= value <= Constants.INT64_MAX:
        exact = int(value)
        return exact
    raise ValueError(f"Decimal {value} has no exact Memgraph integer or float representation")


def name_row_cells(row_cells, column_names):
    computed_return_value = {
        column: (value if not isinstance(value, Decimal) else convert_decimal(value))
        for column, value in zip(column_names, row_cells, strict=False)
    }
    return computed_return_value


def name_row_cells_mysql(row_cells, column_names):
    """
    Convert MySQL row cells to Memgraph-compatible types.
    Handles MySQL-specific types that might cause PyObject conversion errors.
    """
    computed_return_value = {column: convert_mysql_value(value) for column, value in zip(column_names, row_cells, strict=False)}
    return computed_return_value


def convert_mysql_value(value: object) -> object:
    """
    Convert a MySQL value to a Memgraph-compatible type.
    SQL NULL stays null; a value with no string form is refused.
    """
    if value is None:
        return value

    # Handle Decimal types
    if isinstance(value, Decimal):
        computed_return_value = convert_decimal(value)
        return computed_return_value
    # Handle datetime types
    if isinstance(value, (datetime_datetime, datetime_date, datetime_time)):
        # Use ISO 8601 format for consistency
        computed_return_value = value.isoformat()
        return computed_return_value
    # Handle timedelta
    if isinstance(value, datetime_timedelta):
        computed_return_value = str(value)
        return computed_return_value

    # Handle binary data (BLOB, BINARY, VARBINARY)
    if isinstance(value, (bytes, bytearray)):
        try:
            # Try to decode as UTF-8 string first
            computed_return_value = value.decode("utf-8")
            return computed_return_value
        except UnicodeDecodeError:
            # If not valid UTF-8, convert to base64 string
            computed_return_value = base64_b64encode(value).decode("ascii")
            return computed_return_value

    # Handle geometry types (convert to string representation)
    if hasattr(value, "__class__") and "geometry" in str(value.__class__).lower():
        computed_return_value = str(value) if value else ""
        return computed_return_value

    # Handle MySQL-specific numeric types
    if isinstance(value, (int, float, bool)):
        return value

    # Handle string types
    if isinstance(value, str):
        return value

    # Handle list/array types
    if isinstance(value, (list, tuple)):
        computed_return_value = [convert_mysql_value(item) for item in value]
        return computed_return_value

    # Handle dictionary/map types
    if isinstance(value, dict):
        computed_return_value = {k: convert_mysql_value(v) for k, v in value.items()}
        return computed_return_value

    # Any other type migrates as its string representation
    try:
        str_value = str(value)
    except (ValueError, TypeError) as err:
        raise TypeError(f"MySQL value of type {type(value).__name__} has no Memgraph representation") from err
    return str_value


def convert_row_types(row_cells):
    computed_return_value = {
        column: (value if not isinstance(value, Decimal) else convert_decimal(value)) for column, value in row_cells.items()
    }
    return computed_return_value


def check_params_type(params: object, types=(dict, list, tuple)) -> bool:
    if not isinstance(params, types):
        raise TypeError(
            "Database query parameter values must be passed in a container of type List[Any] (or Map, if "
            "migrating from MySQL, Oracle DB or ServiceNow)"
        )
    return False


def convert_neo4j_value(value):
    """Convert Neo4j values to Python-compatible formats; null stays null."""
    if value is None:
        return value

    # Handle Neo4j DateTime objects
    if isinstance(value, Neo4jDateTime) or isinstance(value, Neo4jDate):
        computed_return_value = value.to_native()
        return computed_return_value

    # Handle lists and dicts recursively
    if isinstance(value, list):
        computed_return_value = [convert_neo4j_value(item) for item in value]
        return computed_return_value

    if isinstance(value, dict):
        computed_return_value = {key: convert_neo4j_value(val) for key, val in value.items()}
        return computed_return_value

    # For other types, return as is
    return value


def convert_neo4j_record(record):
    """Convert a Neo4j record to a Python dict with proper type conversion."""
    computed_return_value = {key: convert_neo4j_value(value) for key, value in record.items()}
    return computed_return_value


def build_neo4j_uri(config: dict) -> str:
    host = config.get(Constants.HOST, "localhost")
    port = config.get(Constants.PORT, 7687)
    uri_scheme = config.get(Constants.URI_SCHEME, "bolt")
    computed_return_value = f"{uri_scheme}://{host}:{port}"
    return computed_return_value
