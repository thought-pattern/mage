"""
This module tests modules from this folder one by one by comparing structure after query
is executed on Neo4j and Memgraph. Be sure to have Neo4j and Memgraph instance running.
"""

from logging import INFO as logging_INFO
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger
from os import path as os_path
from pathlib import Path

from gqlalchemy import Memgraph
from neo4j import BoltDriver as neo4j_BoltDriver
from neo4j import GraphDatabase as neo4j_GraphDatabase
from pytest import fixture as pytest_fixture
from pytest import mark as pytest_mark
from pytest import param as pytest_param
from query_neo_mem import (
    Graph,
    clean_memgraph_db,
    clean_neo4j_db,
    execute_query_neo4j,
    mg_execute_cyphers,
    mg_get_graph,
    neo4j_execute_cyphers,
    neo4j_get_graph,
    parse_mem,
    parse_neo4j,
    run_memgraph_query,
    run_neo4j_query,
)
from yaml import Loader as yaml_Loader
from yaml import load as yaml_load

logging_basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging_getLogger("e2e_correctness")
logger.setLevel(logging_INFO)


class TestConstants:
    ABSOLUTE_TOLERANCE = 1e-3

    EXCEPTION = "exception"
    INPUT_FILE = "input.cyp"
    OUTPUT = "output"
    QUERY = "query"
    TEST_MODULE_DIR_SUFFIX = "_test"
    TEST_GROUP_DIR_SUFFIX = "_group"

    ONLINE_TEST_E2E_SETUP = "setup"
    ONLINE_TEST_E2E_CLEANUP = "cleanup"
    ONLINE_TEST_E2E_INPUT_QUERIES = "queries"
    TEST_SUBDIR_PREFIX = "test"
    TEST_FILE = "test.yml"
    MEMGRAPH_QUERY = "memgraph_query"
    NEO4J_QUERY = "neo4j_query"
    CONFIG_FILE = "config.yml"


def get_all_tests():
    """
    Fetch all the tests in the testing folders, and prepare them for execution
    """
    tests = []

    test_path = Path().cwd()

    for module_test_dir in test_path.iterdir():
        if not module_test_dir.is_dir() or not module_test_dir.name.endswith(TestConstants.TEST_MODULE_DIR_SUFFIX):
            continue

        for test_or_group_dir in module_test_dir.iterdir():
            if not test_or_group_dir.is_dir():
                continue

            if test_or_group_dir.name.endswith(TestConstants.TEST_GROUP_DIR_SUFFIX):
                for test_dir in test_or_group_dir.iterdir():
                    if not test_dir.is_dir():
                        continue

                    tests.append(
                        pytest_param(
                            test_dir,
                            id=f"{module_test_dir.stem}-{test_or_group_dir.stem}-{test_dir.stem}",
                        )
                    )
            else:
                tests.append(
                    pytest_param(
                        test_or_group_dir,
                        id=f"{module_test_dir.stem}-{test_or_group_dir.stem}",
                    )
                )
    return tests


tests = get_all_tests()


def load_yaml(path: Path) -> dict:
    """
    Load YAML based file in Python dictionary.
    """
    file_handle = path.open("r")
    computed_return_value = yaml_load(file_handle, Loader=yaml_Loader)
    return computed_return_value


def graphs_equal(memgraph_graph: Graph, neo4j_graph: Graph) -> bool:
    # Each getter sorts its whole graph, so take every canonical snapshot once per comparison.
    memgraph_vertices, neo4j_vertices = memgraph_graph.vertices, neo4j_graph.vertices
    memgraph_edges, neo4j_edges = memgraph_graph.edges, neo4j_graph.edges

    assert len(memgraph_vertices) == len(
        neo4j_vertices
    ), f"The number of vertices is not equal: \
        Memgraph contains {memgraph_vertices} and Neo4j contains {neo4j_vertices}"

    assert len(memgraph_edges) == len(
        neo4j_edges
    ), f"The number of edges is not equal: \
        Memgraph contains {memgraph_edges} and Neo4j contains {neo4j_edges}"

    for mem_vertex, neo_vertex in zip(memgraph_vertices, neo4j_vertices, strict=True):
        if mem_vertex != neo_vertex:
            logger.debug(
                f"The vertices are different: \
            Neo4j vertex: {neo_vertex}\
            Memgraph vertex: {mem_vertex}"
            )
            return False
    for mem_edge, neo_edge in zip(memgraph_edges, neo4j_edges, strict=True):
        if neo_edge != mem_edge:
            logger.debug(
                f"The edges are different: \
                Neo4j edge: {neo_edge}\
                Memgraph edge: {mem_edge}"
            )
            return False
    return True


def validated_queries(test_dict: dict) -> tuple[str, str]:
    """Return the Memgraph and Neo4j queries a test file must declare, failing when either is missing."""
    missing = [key for key in (TestConstants.MEMGRAPH_QUERY, TestConstants.NEO4J_QUERY) if key not in test_dict]
    assert not missing, f"Test file declares no {', '.join(missing)}"
    queries = (test_dict.get(TestConstants.MEMGRAPH_QUERY, ""), test_dict.get(TestConstants.NEO4J_QUERY, ""))
    return queries


def run_test(test_dir: Path, memgraph_db: Memgraph, neo4j_driver: neo4j_BoltDriver) -> bool:
    """
    Run input queries on Memgraph and Neo4j and compare graphs after running test query
    """
    input_cyphers = test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    mg_execute_cyphers(input_cyphers, memgraph_db)
    logger.info(f"Imported data into Memgraph from {input_cyphers}")
    neo4j_execute_cyphers(input_cyphers, neo4j_driver)
    logger.info(f"Imported data into Neo4j from {input_cyphers}")

    test_dict = load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    logger.info(f"Test dict {test_dict}")
    memgraph_query, neo4j_query = validated_queries(test_dict)

    logger.info(f"Running query against Memgraph: {memgraph_query}")
    run_memgraph_query(memgraph_query, memgraph_db)
    logger.info("Done")

    logger.info(f"Running query against Neo4j: {neo4j_query}")
    run_neo4j_query(neo4j_query, neo4j_driver)
    logger.info("Done")

    mg_graph = mg_get_graph(memgraph_db)
    neo4j_graph = neo4j_get_graph(neo4j_driver)

    assert graphs_equal(mg_graph, neo4j_graph), "The graphs are not equal, check the logs for more details"
    return False


def run_path_test(test_dir: Path, memgraph_db: Memgraph, neo4j_driver: neo4j_BoltDriver) -> bool:
    """
    Run input queries on Memgraph and Neo4j and compare path results after running test query
    """
    input_cyphers = test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    logger.info(f"Importing data from {input_cyphers}")
    mg_execute_cyphers(input_cyphers, memgraph_db)
    logger.info("Imported data into Memgraph")
    neo4j_execute_cyphers(input_cyphers, neo4j_driver)
    logger.info("Imported data into Neo4j")

    test_dict = load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    logger.info(f"Test dict {test_dict}")
    memgraph_query, neo4j_query = validated_queries(test_dict)

    logger.info(f"Running query against Memgraph: {memgraph_query}")
    memgraph_results = memgraph_db.execute_and_fetch(memgraph_query)
    memgraph_paths = parse_mem(memgraph_results)
    logger.info("Done")

    logger.info(f"Running query against Neo4j: {neo4j_query}")
    neo4j_results = execute_query_neo4j(neo4j_driver, neo4j_query)
    neo4j_paths = parse_neo4j(neo4j_results)
    logger.info("Done")

    assert memgraph_paths == neo4j_paths
    return False


def check_path_option(test_dir):
    config_path = test_dir.joinpath(TestConstants.CONFIG_FILE)
    if os_path.exists(config_path):
        config_dict = load_yaml(config_path)
        if "path_option" in config_dict:
            option = config_dict.get("path_option", "").strip()
            computed_return_value = option == "True"
            return computed_return_value
    return False


@pytest_fixture(scope="session")
def memgraph_port(pytestconfig):
    computed_return_value = pytestconfig.getoption("--memgraph-port")
    return computed_return_value


@pytest_fixture(scope="session", autouse=True)
def memgraph_db(memgraph_port):
    # One client serves the whole session; GQLAlchemy opens and reuses its connection on demand.
    memgraph_db = Memgraph("localhost", memgraph_port)
    logger.info("Created Memgraph connection")

    yield memgraph_db


@pytest_fixture(scope="session")
def neo4j_port(pytestconfig):
    computed_return_value = pytestconfig.getoption("--neo4j-port")
    return computed_return_value


@pytest_fixture(scope="session", autouse=True)
def neo4j_driver(neo4j_port):
    neo4j_driver = neo4j_GraphDatabase.driver(f"bolt://localhost:{neo4j_port}", encrypted=False)
    logger.info("Created neo4j driver")

    # The session owns this driver's connection pool, so release it even when teardown is interrupted.
    try:
        yield neo4j_driver
    finally:
        neo4j_driver.close()


@pytest_mark.parametrize("test_dir", tests)
def test_end2end(
    test_dir: Path,
    memgraph_db: Memgraph,
    neo4j_driver: neo4j_BoltDriver,
):
    logger.debug("Dropping the Memgraph and Neo4j databases.")

    clean_memgraph_db(memgraph_db)
    clean_neo4j_db(neo4j_driver)

    if test_dir.name.startswith(TestConstants.TEST_SUBDIR_PREFIX):
        if check_path_option(test_dir):
            run_path_test(test_dir, memgraph_db, neo4j_driver)
        else:
            run_test(test_dir, memgraph_db, neo4j_driver)
    else:
        logger.info(f"Skipping directory: {test_dir.name}")

    # Clean database once testing module is finished
    clean_memgraph_db(memgraph_db)
    clean_neo4j_db(neo4j_driver)
