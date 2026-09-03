"""Utilities for migrate."""

from base64 import b64encode as base64_b64encode
from csv import reader as csv_reader_2
from datetime import date as datetime_date
from datetime import datetime as datetime_datetime
from datetime import time as datetime_time
from datetime import timedelta as datetime_timedelta
from decimal import Decimal
from hashlib import sha256 as hashlib_sha256
from io import BytesIO as io_BytesIO
from io import TextIOWrapper as io_TextIOWrapper
from json import dumps as json_dumps
from json import load as json_load
from os import getenv as os_getenv
from re import match as re_match

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
    COLUMN_NAMES = "column_names"
    CONNECTION = "connection"
    CURSOR = "cursor"
    DATABASE = "database"
    DRIVER = "driver"
    HOST = "host"
    I_COLUMN_NAME = 0
    PASSWORD = "password"
    PORT = "port"
    RESULT = "result"
    SESSION = "session"
    URI_SCHEME = "uri_scheme"
    USERNAME = "username"


class NullConnection:
    """Stable no-op database connection used for missing cache entries."""

    def commit(self):
        return False

    def close(self):
        return False


class NullCursor:
    """Stable empty cursor used for missing cache entries."""

    description: tuple = ()

    def fetchmany(self, size):
        del size
        return []


NULL_CONNECTION = NullConnection()
NULL_CURSOR = NullCursor()


def get_query_hash(query: str, config: mgp_Map, params: mgp_Nullable[mgp_Any] = False) -> str:
    """
    Create a hash from query, config, and params to use as a cache key.

    :param query: The query string (or table name, endpoint, file path, etc.)
    :param config: Configuration map
    :param params: Optional query parameters
    """
    if params is None:
        params = False
    config_dict = dict(config)
    config_str = json_dumps(config_dict, sort_keys=True, default=str)

    params_str = ""
    if params is not False:
        if isinstance(params, dict):
            params_str = json_dumps(params, sort_keys=True, default=str)
        elif isinstance(params, (list, tuple)):
            params_str = json_dumps(list(params), sort_keys=False, default=str)
        else:
            params_str = str(params)

    hash_input = f"{query}|{config_str}|{params_str}"
    computed_return_value = hashlib_sha256(hash_input.encode("utf-8")).hexdigest()
    return computed_return_value


# MYSQL

mysql_dict = {}


def init_migrate_mysql(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    global mysql_dict

    if params:
        check_params_type(params)
    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    query_hash = get_query_hash(table_or_sql, config, params)

    # check if query is already running
    if query_hash in mysql_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    mysql_dict[query_hash] = {}

    connection = mysql_connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(table_or_sql, params=params)

    mysql_dict.get(query_hash, {})[Constants.CONNECTION] = connection
    mysql_dict.get(query_hash, {})[Constants.CURSOR] = cursor
    mysql_dict.get(query_hash, {})[Constants.COLUMN_NAMES] = [column[Constants.I_COLUMN_NAME] for column in cursor.description]
    return False


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
    global mysql_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    query_hash = get_query_hash(table_or_sql, config, params)
    cursor = mysql_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    column_names = mysql_dict.get(query_hash, {}).get(Constants.COLUMN_NAMES, [])

    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    result = [mgp_Record(row=name_row_cells_mysql(row, column_names)) for row in rows]

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not result:
        cleanup_mysql_by_hash(query_hash)

    return result


def cleanup_mysql_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global mysql_dict

    if query_hash in mysql_dict:
        mysql_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).commit()
        mysql_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).close()
        mysql_dict.pop(query_hash, {})
    return False


def cleanup_migrate_mysql():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(mysql, init_migrate_mysql, cleanup_migrate_mysql)

# SQL SERVER

sql_server_dict = {}


def init_migrate_sql_server(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    global sql_server_dict

    if params:
        check_params_type(params, (list, tuple))
    else:
        params = []

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    query_hash = get_query_hash(table_or_sql, config, params)

    # check if query is already running
    if query_hash in sql_server_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    sql_server_dict[query_hash] = {}

    connection = pyodbc_connect(**config)
    cursor = connection.cursor()
    cursor.execute(table_or_sql, *params)

    sql_server_dict.get(query_hash, {})[Constants.CONNECTION] = connection
    sql_server_dict.get(query_hash, {})[Constants.CURSOR] = cursor
    sql_server_dict.get(query_hash, {})[Constants.COLUMN_NAMES] = [column[Constants.I_COLUMN_NAME] for column in cursor.description]
    return False


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
    global sql_server_dict

    if not params:
        params = []

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    query_hash = get_query_hash(table_or_sql, config, params)
    cursor = sql_server_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    column_names = sql_server_dict.get(query_hash, {}).get(Constants.COLUMN_NAMES, [])
    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    result = [mgp_Record(row=name_row_cells(row, column_names)) for row in rows]

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not result:
        cleanup_sql_server_by_hash(query_hash)

    return result


def cleanup_sql_server_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global sql_server_dict

    if query_hash in sql_server_dict:
        sql_server_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).commit()
        sql_server_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).close()
        sql_server_dict.pop(query_hash, {})
    return False


def cleanup_migrate_sql_server():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(sql_server, init_migrate_sql_server, cleanup_migrate_sql_server)

# Oracle DB

oracle_db_dict = {}


def init_migrate_oracle_db(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    global oracle_db_dict

    if params:
        check_params_type(params)

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql}"

    if not config:
        config = {}

    # To prevent query execution from hanging
    config["disable_oob"] = True

    query_hash = get_query_hash(table_or_sql, config, params)

    # check if query is already running
    if query_hash in oracle_db_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    oracle_db_dict[query_hash] = {}

    connection = oracledb_connect(**config)
    cursor = connection.cursor()

    if not params:
        cursor.execute(table_or_sql)
    elif isinstance(params, (list, tuple)):
        cursor.execute(table_or_sql, params)
    else:
        cursor.execute(table_or_sql, **params)

    oracle_db_dict.get(query_hash, {})[Constants.CONNECTION] = connection
    oracle_db_dict.get(query_hash, {})[Constants.CURSOR] = cursor
    oracle_db_dict.get(query_hash, {})[Constants.COLUMN_NAMES] = [column[Constants.I_COLUMN_NAME] for column in cursor.description]
    return False


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

    global oracle_db_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql}"

    if not config:
        config = {}
    config["disable_oob"] = True

    query_hash = get_query_hash(table_or_sql, config, params)
    cursor = oracle_db_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    column_names = oracle_db_dict.get(query_hash, {}).get(Constants.COLUMN_NAMES, [])
    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    result = [mgp_Record(row=name_row_cells(row, column_names)) for row in rows]

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not result:
        cleanup_oracle_db_by_hash(query_hash)

    return result


def cleanup_oracle_db_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global oracle_db_dict

    if query_hash in oracle_db_dict:
        oracle_db_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).commit()
        oracle_db_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).close()
        oracle_db_dict.pop(query_hash, {})
    return False


def cleanup_migrate_oracle_db():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(oracle_db, init_migrate_oracle_db, cleanup_migrate_oracle_db)


# PostgreSQL dictionary to store connections and cursors by thread
postgres_dict = {}


def init_migrate_postgresql(
    table_or_sql: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    global postgres_dict

    if params:
        check_params_type(params, (list, tuple))
    else:
        params = []

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    query_hash = get_query_hash(table_or_sql, config, params)

    # check if query is already running
    if query_hash in postgres_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    postgres_dict[query_hash] = {}

    connection = psycopg2_connect(**config)
    cursor = connection.cursor()
    cursor.execute(table_or_sql, params)

    postgres_dict.get(query_hash, {})[Constants.CONNECTION] = connection
    postgres_dict.get(query_hash, {})[Constants.CURSOR] = cursor
    postgres_dict.get(query_hash, {})[Constants.COLUMN_NAMES] = [column.name for column in cursor.description]
    return False


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
    global postgres_dict

    if not params:
        params = []

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    if query_is_table(table_or_sql):
        table_or_sql = f"SELECT * FROM {table_or_sql};"

    query_hash = get_query_hash(table_or_sql, config, params)
    cursor = postgres_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    column_names = postgres_dict.get(query_hash, {}).get(Constants.COLUMN_NAMES, [])

    rows = cursor.fetchmany(Constants.BATCH_SIZE)

    result = [mgp_Record(row=name_row_cells(row, column_names)) for row in rows]

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not result:
        cleanup_postgresql_by_hash(query_hash)

    return result


def cleanup_postgresql_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global postgres_dict

    if query_hash in postgres_dict:
        postgres_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).commit()
        postgres_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).close()
        postgres_dict.pop(query_hash, {})
    return False


def cleanup_migrate_postgresql():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(postgresql, init_migrate_postgresql, cleanup_migrate_postgresql)


# S3
s3_dict = {}


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
    global s3_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    # Extract S3 bucket and key
    if not file_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Expected 's3://bucket-name/path'.")

    file_path_no_protocol = file_path[5:]
    bucket_name, *key_parts = file_path_no_protocol.split("/")
    s3_key = "/".join(key_parts)

    query_hash = get_query_hash(file_path, config)

    # check if query is already running
    if query_hash in s3_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    # Initialize S3 client
    s3_client = boto3_client(
        "s3",
        aws_access_key_id=config.get("aws_access_key_id", os_getenv("AWS_ACCESS_KEY_ID", False)),
        aws_secret_access_key=config.get("aws_secret_access_key", os_getenv("AWS_SECRET_ACCESS_KEY", False)),
        aws_session_token=config.get("aws_session_token", os_getenv("AWS_SESSION_TOKEN", False)),
        region_name=config.get("region_name", os_getenv("AWS_REGION", False)),
    )

    # Fetch and read file as a streaming object
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    # Convert binary stream to text stream
    text_stream = io_TextIOWrapper(response.get("Body", io_BytesIO()), encoding="utf-8")

    # Read CSV headers
    csv_reader = csv_reader_2(text_stream)
    column_names = next(csv_reader)  # First row contains column names

    s3_dict[query_hash] = {}
    s3_dict.get(query_hash, {})[Constants.CURSOR] = csv_reader
    s3_dict.get(query_hash, {})[Constants.COLUMN_NAMES] = column_names
    return False


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
    global s3_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query_hash = get_query_hash(file_path, config)
    csv_reader = s3_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    column_names = s3_dict.get(query_hash, {}).get(Constants.COLUMN_NAMES, [])

    batch_rows = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = next(csv_reader)
            batch_rows.append(mgp_Record(row=name_row_cells(row, column_names)))
        except StopIteration:
            break

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not batch_rows:
        cleanup_s3_by_hash(query_hash)

    return batch_rows


def cleanup_s3_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global s3_dict

    if query_hash in s3_dict:
        s3_dict.pop(query_hash, False)
    return False


def cleanup_migrate_s3():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(s3, init_migrate_s3, cleanup_migrate_s3)


neo4j_dict = {}


def init_migrate_neo4j(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    if params is None:
        params = False
    global neo4j_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query = formulate_cypher_query(label_or_rel_or_query)
    query_hash = get_query_hash(query, config, params)

    # check if query is already running
    if query_hash in neo4j_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    uri = build_neo4j_uri(config)
    username = config.get(Constants.USERNAME, "neo4j")
    password = config.get(Constants.PASSWORD, "password")
    database = config.get(Constants.DATABASE, False)  # None means default database

    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Create session with optional database parameter
    if database:
        session = driver.session(database=database)
    else:
        session = driver.session()

    # Neo4j expects params to be a dict or None
    cypher_params = params if params is not False else {}
    run_query = getattr(session, "run", False)
    if not callable(run_query):
        raise TypeError("Neo4j session must provide a callable run operation")
    result = run_query(query, parameters=cypher_params)

    neo4j_dict[query_hash] = {}
    neo4j_dict.get(query_hash, {})[Constants.DRIVER] = driver
    neo4j_dict.get(query_hash, {})[Constants.SESSION] = session
    neo4j_dict.get(query_hash, {})[Constants.RESULT] = result
    return False


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
    global neo4j_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query = formulate_cypher_query(label_or_rel_or_query)
    query_hash = get_query_hash(query, config, params)
    result = neo4j_dict.get(query_hash, {}).get(Constants.RESULT, {})

    # Fetch up to BATCH_SIZE records
    batch = []
    for record in result:
        # Convert neo4j.Record to dict with proper type conversion
        batch.append(mgp_Record(row=convert_neo4j_record(record)))

        # Check if we've reached the batch size limit
        if len(batch) >= Constants.BATCH_SIZE:
            break

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not batch:
        cleanup_neo4j_by_hash(query_hash)

    return batch


def cleanup_neo4j_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global neo4j_dict

    if query_hash in neo4j_dict:
        session = neo4j_dict.get(query_hash, {}).get(Constants.SESSION, False)
        driver = neo4j_dict.get(query_hash, {}).get(Constants.DRIVER, False)
        if session:
            session.close()
        if driver:
            driver.close()
        neo4j_dict.pop(query_hash, False)
    return False


def cleanup_migrate_neo4j():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(neo4j, init_migrate_neo4j, cleanup_migrate_neo4j)


# Dictionary to store Flight connections per thread
flight_dict = {}


def init_migrate_arrow_flight(
    query: str,
    config: mgp_Map,
    config_path: str = "",
):
    global flight_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query_hash = get_query_hash(query, config)

    # check if query is already running
    if query_hash in flight_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    host = config.get(Constants.HOST, False)
    port = config.get(Constants.PORT, False)
    username = config.get(Constants.USERNAME, "")
    password = config.get(Constants.PASSWORD, "")

    # Encode credentials
    auth_string = f"{username}:{password}".encode("utf-8")
    encoded_auth = base64_b64encode(auth_string).decode("utf-8")

    # Establish Flight connection
    client = flight.connect(f"grpc://{host}:{port}")

    # Authenticate
    options = flight.FlightCallOptions(headers=[(b"authorization", f"Basic {encoded_auth}".encode("utf-8"))])

    flight_info = client.get_flight_info(flight.FlightDescriptor.for_command(query), options)

    flight_dict[query_hash] = {}
    flight_dict.get(query_hash, {})[Constants.CONNECTION] = client
    flight_dict.get(query_hash, {})[Constants.CURSOR] = iter(fetch_flight_data(client, flight_info, options))
    return False


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
    :param config: Arrow Flight connection configuration
    :param config_path: Path to a JSON config file
    :return: Stream of rows from Arrow Flight
    """
    global flight_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query_hash = get_query_hash(query, config)
    cursor = flight_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    batch = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = convert_row_types(next(cursor))
            batch.append(mgp_Record(row=row))
        except StopIteration:
            break

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not batch:
        cleanup_arrow_flight_by_hash(query_hash)

    return batch


def cleanup_arrow_flight_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global flight_dict

    if query_hash in flight_dict:
        flight_dict.pop(query_hash, False)
    return False


def cleanup_migrate_arrow_flight():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(arrow_flight, init_migrate_arrow_flight, cleanup_migrate_arrow_flight)


# Dictionary to store DuckDB connections and cursors per thread
duckdb_dict = {}


def init_migrate_duckdb(query: str, setup_queries: mgp_Nullable[list[str]] = False):
    """
    Initialize an in-memory DuckDB connection and execute the query.

    :param query: SQL query to execute
    :param setup_queries: Optional list of setup queries to execute before the main query
    """
    if setup_queries is None:
        setup_queries = False
    global duckdb_dict

    # Create hash from query and setup_queries
    setup_queries_str = json_dumps(setup_queries, sort_keys=False) if setup_queries else ""
    query_hash = hashlib_sha256(f"{query}|{setup_queries_str}".encode("utf-8")).hexdigest()

    # check if query is already running
    if query_hash in duckdb_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    # Ensure a fresh in-memory DuckDB instance for each query
    connection = duckDB_connect()
    cursor = connection.cursor()
    if setup_queries is not False:
        for setup_query in setup_queries:
            cursor.execute(setup_query)

    cursor.execute(query)

    duckdb_dict[query_hash] = {}
    duckdb_dict.get(query_hash, {})[Constants.CONNECTION] = connection
    duckdb_dict.get(query_hash, {})[Constants.CURSOR] = cursor
    duckdb_dict.get(query_hash, {})[Constants.COLUMN_NAMES] = [desc[0] for desc in cursor.description]
    return False


def duckdb(query: str, setup_queries: mgp_Nullable[list[str]] = False) -> list[mgp_Record]:
    """
    Fetch rows from DuckDB in batches.

    :param query: SQL query to execute
    :param setup_queries: Optional list of setup queries to execute before the main query
    :return: The result table as a stream of rows
    """
    global duckdb_dict

    setup_queries_str = json_dumps(setup_queries, sort_keys=False) if setup_queries else ""
    query_hash = hashlib_sha256(f"{query}|{setup_queries_str}".encode("utf-8")).hexdigest()
    cursor = duckdb_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)
    column_names = duckdb_dict.get(query_hash, {}).get(Constants.COLUMN_NAMES, [])

    rows = cursor.fetchmany(Constants.BATCH_SIZE)
    result = [mgp_Record(row=name_row_cells(row, column_names)) for row in rows]

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not result:
        cleanup_duckdb_by_hash(query_hash)

    return result


def cleanup_duckdb_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global duckdb_dict

    if query_hash in duckdb_dict:
        if Constants.CONNECTION in duckdb_dict.get(query_hash, {}):
            duckdb_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).close()
        duckdb_dict.pop(query_hash, False)
    return False


def cleanup_migrate_duckdb():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(duckdb, init_migrate_duckdb, cleanup_migrate_duckdb)


memgraph_dict = {}


def init_migrate_memgraph(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    global memgraph_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query = formulate_cypher_query(label_or_rel_or_query)
    query_hash = get_query_hash(query, config, params)

    # check if query is already running
    if query_hash in memgraph_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    memgraph_db = Memgraph(**config)
    cursor = memgraph_db.execute_and_fetch(query, params)

    memgraph_dict[query_hash] = {}
    memgraph_dict.get(query_hash, {})[Constants.CONNECTION] = memgraph_db
    memgraph_dict.get(query_hash, {})[Constants.CURSOR] = cursor
    return False


def memgraph(
    label_or_rel_or_query: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    (
        "\n    Migrate data from Memgraph to another Memgraph instance. Can migrate a specific node la"  # Continue literal.
        "bel, relationship type, or execute a custom Cypher query.\n\n    :param label_or_rel_or_query:"  # Continue literal.
        " Node label, relationship type, or a Cypher query\n    :param config: Connection configuratio"  # Continue literal.
        "n for Memgraph\n    :param config_path: Path to a JSON file containing connection parameters\n"  # Continue literal.
        "    :param params: Optional query parameters\n    :return: Stream of rows from Memgraph\n"
    )
    global memgraph_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query = formulate_cypher_query(label_or_rel_or_query)
    query_hash = get_query_hash(query, config, params)
    cursor = memgraph_dict.get(query_hash, {}).get(Constants.CURSOR, NULL_CURSOR)

    result = [mgp_Record(row=row) for row in (next(cursor, False) for _ in range(Constants.BATCH_SIZE)) if row is not None]

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not result:
        cleanup_memgraph_by_hash(query_hash)

    return result


def cleanup_memgraph_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global memgraph_dict

    if query_hash in memgraph_dict:
        if Constants.CONNECTION in memgraph_dict.get(query_hash, {}):
            memgraph_dict.get(query_hash, {}).get(Constants.CONNECTION, NULL_CONNECTION).close()
        memgraph_dict.pop(query_hash, False)
    return False


def cleanup_migrate_memgraph():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(memgraph, init_migrate_memgraph, cleanup_migrate_memgraph)


servicenow_dict = {}


def init_migrate_servicenow(
    endpoint: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
):
    """
    Initialize the connection to the ServiceNow REST API and fetch the JSON data.

    :param endpoint: ServiceNow API endpoint (full URL)
    :param config: Configuration map containing authentication details (username, password, instance URL, etc.)
    :param config_path: Optional path to a JSON file containing authentication details
    :param params: Optional query parameters for filtering results
    """
    global servicenow_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query_hash = get_query_hash(endpoint, config, params)

    # check if query is already running
    if query_hash in servicenow_dict:
        raise RuntimeError(
            "Migrate module with these parameters is already running. Please wait for it to finish before starting a new one."
        )

    auth = (
        config.get(Constants.USERNAME, False),
        config.get(Constants.PASSWORD, False),
    )
    headers = {"Accept": "application/json"}

    response = requests_get(endpoint, auth=auth, headers=headers, params=params)
    response.raise_for_status()

    data = response.json().get(Constants.RESULT, [])
    if not data:
        raise ValueError("No data found in ServiceNow response")

    servicenow_dict[query_hash] = {}
    servicenow_dict.get(query_hash, {})[Constants.CURSOR] = iter(data)
    return False


def servicenow(
    endpoint: str,
    config: mgp_Map,
    config_path: str = "",
    params: mgp_Nullable[mgp_Any] = False,
) -> list[mgp_Record]:
    """
    Fetch rows from the ServiceNow REST API in batches.

    :param endpoint: ServiceNow API endpoint (full URL)
    :param config: Authentication details (username, password, instance URL, etc.)
    :param config_path: Optional path to a JSON file containing authentication details
    :param params: Optional query parameters for filtering results
    :return: The result data as a stream of rows
    """
    global servicenow_dict

    if len(config_path) > 0:
        config = combine_config(config=config, config_path=config_path)

    query_hash = get_query_hash(endpoint, config, params)
    data_iter = servicenow_dict.get(query_hash, {}).get(Constants.CURSOR, {})

    batch_rows = []
    for _ in range(Constants.BATCH_SIZE):
        try:
            row = next(data_iter)
            batch_rows.append(mgp_Record(row=row))
        except StopIteration:
            break

    # if results are empty, cleanup the query since cleanup doesn't accept any parameters
    if not batch_rows:
        cleanup_servicenow_by_hash(query_hash)

    return batch_rows


def cleanup_servicenow_by_hash(query_hash: str):
    """Internal cleanup function that takes a query hash."""
    global servicenow_dict

    if query_hash in servicenow_dict:
        servicenow_dict.pop(query_hash, False)
    return False


def cleanup_migrate_servicenow():
    """Cleanup function called by mgp framework (no parameters)."""
    return False


mgp_add_batch_read_proc(servicenow, init_migrate_servicenow, cleanup_migrate_servicenow)


def formulate_cypher_query(label_or_rel_or_query: str) -> str:
    words = label_or_rel_or_query.split()
    if len(words) > 1:
        return label_or_rel_or_query  # Treat it as a Cypher query if multiple words exist

    # Try to see if the syntax matches similar to (:Label) to migrate only nodes
    node_match = re_match(r"^\(\s*:(\w+)\s*\)$", label_or_rel_or_query)

    # Try to see if the syntax matches similar to [:REL_TYPE] to migrate only relationships
    rel_match = re_match(r"^\[\s*:(\w+)\s*\]$", label_or_rel_or_query)

    if node_match:
        label = node_match.group(1)
        computed_return_value = f"MATCH (n:{label}) RETURN labels(n) as labels, properties(n) as properties"
        return computed_return_value

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
    computed_return_value = len(table_or_sql.split()) == 1
    return computed_return_value


def combine_config(config: mgp_Map, config_path: str) -> dict[str, object]:
    assert len(config_path), "Path must not be empty"

    file_config = {}
    try:
        with open(config_path, "r") as file:
            file_config = json_load(file)
    except Exception as caught_error_1202:
        raise OSError("Could not open/read file.") from caught_error_1202

    config.update(file_config)
    return config


def name_row_cells(row_cells, column_names):
    computed_return_value = {
        column: (value if not isinstance(value, Decimal) else float(value))
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
    Returns ``False`` for unsupported types and logs a warning.
    """
    if value is None:
        return False

    # Handle Decimal types
    if isinstance(value, Decimal):
        computed_return_value = float(value)
        return computed_return_value
    # Handle datetime types
    if isinstance(value, (datetime_datetime, datetime_date, datetime_time)):
        # Use ISO 8601 format for consistency
        try:
            computed_return_value = value.isoformat()
            return computed_return_value
        except Exception:
            computed_return_value = str(value)
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

    # For any other unsupported types, convert to string or return None
    try:
        # Try to convert to string
        str_value = str(value)
        return str_value
    except (ValueError, TypeError):
        # If string conversion fails, return None
        return False


def convert_row_types(row_cells):
    computed_return_value = {
        column: (value if not isinstance(value, Decimal) else float(value)) for column, value in row_cells.items()
    }
    return computed_return_value


def check_params_type(params: object, types=(dict, list, tuple)) -> bool:
    if not isinstance(params, types):
        raise TypeError(
            "Database query parameter values must be passed in a container of type List[Any] (or Map, if "
            "migrating from MySQL or Oracle DB)"
        )
    return False


def convert_neo4j_value(value):
    """Convert Neo4j values to Python-compatible formats."""
    if value is None:
        return False

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


def build_neo4j_uri(config: mgp_Map) -> str:
    host = config.get(Constants.HOST, "localhost")
    port = config.get(Constants.PORT, 7687)
    uri_scheme = config.get(Constants.URI_SCHEME, "bolt")
    computed_return_value = f"{uri_scheme}://{host}:{port}"
    return computed_return_value
