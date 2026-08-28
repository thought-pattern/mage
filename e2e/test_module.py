"""Utilities for test module."""

from importlib import import_module as _import_module
from pathlib import Path
from typing import Dict, List

from gqlalchemy import Memgraph, Node, Relationship
from gqlalchemy import Path as path_gql
from mgclient import DatabaseError as mgclient_DatabaseError
from mgclient import Node as node_mgclient
from mgclient import Relationship as relationship_mgclient
from pytest import approx as pytest_approx
from pytest import fail as pytest_fail
from pytest import fixture as pytest_fixture
from pytest import mark as pytest_mark
from pytest import param as pytest_param
from pytest import raises as pytest_raises
from yaml import Loader as yaml_Loader
from yaml import load as yaml_load

os = _import_module("os")


@pytest_fixture
def db():
    _return_value = Memgraph()
    return _return_value


class TestConstants:
    ABSOLUTE_TOLERANCE = 1e-3

    EXCEPTION = "exception"
    INPUT_FILE = "input.cyp"
    OUTPUT = "output"
    QUERY = "query"
    TEST_FILE = "test.yml"
    FILENAME_PLACEHOLDER = "_file"
    TEST_MODULE_DIR_SUFFIX = "_test"
    TEST_GROUP_DIR_SUFFIX = "_group"

    ONLINE_TEST_E2E_SETUP = "setup"
    ONLINE_TEST_E2E_CLEANUP = "cleanup"
    ONLINE_TEST_E2E_INPUT_QUERIES = "queries"
    ONLINE_TEST_SUBDIR_PREFIX = "test_online"

    EXPORT_TEST_E2E_NODES = "nodes"
    EXPORT_TEST_E2E_RELATIONSHIPS = "relationships"
    EXPORT_TEST_E2E_INPUT_QUERIES = "queries"
    EXPORT_TEST_E2E_EXPORT_QUERY = "export"
    EXPORT_TEST_E2E_IMPORT_QUERY = "import"
    EXPORT_TEST_E2E_PLACEHOLDER_FILENAME = "_exportfile"
    EXPORT_TEST_E2E_OUTPUT_FILE = "/home/memgraph/_exported_data"
    EXPORT_TEST_SUBDIR_PREFIX = "test_export"


def _node_to_dict(data):
    labels = (
        data.labels
        if hasattr(data, "labels")
        else (data._labels if isinstance(data, Node) else [])
    )
    properties = data.properties if hasattr(data, "properties") else data._properties
    _return_value = {"labels": list(labels), "properties": properties}
    return _return_value


def _relationship_to_dict(data):
    label = (
        data.type
        if hasattr(data, "type")
        else (data._type if isinstance(data, Relationship) else "")
    )
    properties = data.properties if hasattr(data, "properties") else data._properties
    return {"label": label, "properties": properties}


def _path_to_dict(data):
    nodes = data.nodes if hasattr(data, "nodes") else data._nodes
    relationships = (
        data.relationships if hasattr(data, "relationships") else data._relationships
    )
    _return_value = {
        "nodes": [_node_to_dict(node) for node in nodes],
        "relationships": [
            _relationship_to_dict(relationship) for relationship in relationships
        ],
    }
    return _return_value


def _replace(data, match_classes):
    if isinstance(data, dict):
        _return_value = {k: _replace(v, match_classes) for k, v in data.items()}
        return _return_value
    elif isinstance(data, list):
        _return_value = [_replace(i, match_classes) for i in data]
        return _return_value
    elif isinstance(data, float):
        _return_value = pytest_approx(data, abs=TestConstants.ABSOLUTE_TOLERANCE)
        return _return_value
    elif isinstance(data, node_mgclient) or isinstance(data, Node):
        _return_value = _node_to_dict(data)
        return _return_value
    elif isinstance(data, relationship_mgclient) or isinstance(data, Relationship):
        _return_value = _relationship_to_dict(data)
        return _return_value
    elif isinstance(data, path_gql):
        _return_value = _path_to_dict(data)
        return _return_value
    else:
        _return_value = _node_to_dict(data) if isinstance(data, match_classes) else data
        return _return_value


def _replace_filename(query: str, dir: Path):
    _return_value = query.replace(
        TestConstants.FILENAME_PLACEHOLDER, "/".join([str(dir), "file"])
    )
    return _return_value


def prepare_tests():
    """
    Fetch all the tests in the testing folders, and prepare them for execution
    """
    tests = []

    test_path = Path().cwd()

    for module_test_dir in test_path.iterdir():
        if not module_test_dir.is_dir() or not module_test_dir.name.endswith(
            TestConstants.TEST_MODULE_DIR_SUFFIX
        ):
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


tests = prepare_tests()


def _load_yaml(path: Path) -> Dict:
    """
    Load YAML based file in Python dictionary.
    """
    file_handle = path.open("r")
    _return_value = yaml_load(file_handle, Loader=yaml_Loader)
    return _return_value


def _execute_cyphers(input_cyphers: List[str], db: Memgraph):
    """
    Execute commands against Memgraph
    """
    for query in input_cyphers:
        db.execute(query)
    return False


def _run_test(test_dict: Dict, db: Memgraph):
    """
    Run queries on Memgraph and compare them to expected results stored in test_dict
    """
    test_query = test_dict.get(TestConstants.QUERY, "")
    output_test = TestConstants.OUTPUT in test_dict
    exception_test = TestConstants.EXCEPTION in test_dict

    if not (output_test ^ exception_test):
        pytest_fail("Test file has no valid format.")

    if output_test:
        result_query = list(db.execute_and_fetch(test_query))

        result = _replace(result_query, Node)

        expected = test_dict.get(TestConstants.OUTPUT, False)

        assert result == expected

    if exception_test:
        with pytest_raises(mgclient_DatabaseError):
            db.execute(test_query)
    return False


def _get_nodes_and_relationships(
    nodes_query: str, relationships_query: str, db: Memgraph
):
    _return_value = (
        list(db.execute_and_fetch(nodes_query)),
        list(db.execute_and_fetch(relationships_query)),
    )
    return _return_value


def _test_export(test_dir: Path, db: Memgraph):
    """
    Testing export modules.
    """

    test_name = test_dir.name
    output_file = f"{TestConstants.EXPORT_TEST_E2E_OUTPUT_FILE}_{test_name}"

    input_dict = _load_yaml(test_dir.joinpath(TestConstants.INPUT_FILE))

    queries = input_dict.get(TestConstants.EXPORT_TEST_E2E_INPUT_QUERIES, False)
    db.execute(queries)

    nodes_query = input_dict.get(TestConstants.EXPORT_TEST_E2E_NODES, False)
    relationships_query = input_dict.get(
        TestConstants.EXPORT_TEST_E2E_RELATIONSHIPS, False
    )

    old_nodes, old_relationships = _get_nodes_and_relationships(
        nodes_query, relationships_query, db
    )

    test_dict = _load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    export_query = test_dict.get(
        TestConstants.EXPORT_TEST_E2E_EXPORT_QUERY, ""
    ).replace(
        TestConstants.EXPORT_TEST_E2E_PLACEHOLDER_FILENAME,
        "".join(["'", output_file, "'"]),
    )
    import_query = test_dict.get(
        TestConstants.EXPORT_TEST_E2E_IMPORT_QUERY, ""
    ).replace(
        TestConstants.EXPORT_TEST_E2E_PLACEHOLDER_FILENAME,
        "".join(["'", output_file, "'"]),
    )

    db.execute(export_query)
    db.execute("MATCH (n) DETACH DELETE n;")
    db.execute(import_query)

    new_nodes, new_relationships = _get_nodes_and_relationships(
        nodes_query, relationships_query, db
    )

    assert _replace(old_nodes, Node) == _replace(new_nodes, Node)
    assert _replace(old_relationships, Relationship) == _replace(
        new_relationships, Relationship
    )
    return False


def _test_static(test_dir: Path, db: Memgraph):
    """
    Testing static modules.
    """
    input_cyphers = [
        _replace_filename(query, test_dir)
        for query in test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    ]
    _execute_cyphers(input_cyphers, db)

    test_dict = _load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    test_dict[TestConstants.QUERY] = _replace_filename(
        test_dict.get(TestConstants.QUERY, False), test_dir
    )
    _run_test(test_dict, db)
    return False


def _test_online(test_dir: Path, db: Memgraph):
    """
    Testing online modules. Checkpoint testing
    """
    checkpoint_input = _load_yaml(test_dir.joinpath(TestConstants.INPUT_FILE))
    checkpoint_test_dicts = _load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))

    setup_cyphers = checkpoint_input.get(TestConstants.ONLINE_TEST_E2E_SETUP, False)
    checkpoint_input_cyphers = checkpoint_input[
        TestConstants.ONLINE_TEST_E2E_INPUT_QUERIES
    ]
    cleanup_cyphers = checkpoint_input.get(TestConstants.ONLINE_TEST_E2E_CLEANUP, False)

    # Run optional setup queries
    if setup_cyphers:
        _execute_cyphers(setup_cyphers.splitlines(), db)

    try:
        # Execute cypher queries and compare them with results
        for input_cyphers_raw, test_dict in zip(
            checkpoint_input_cyphers, checkpoint_test_dicts, strict=False
        ):
            input_cyphers = input_cyphers_raw.splitlines()
            _execute_cyphers(input_cyphers, db)
            _run_test(test_dict, db)
    finally:
        # Run optional cleanup queries
        if cleanup_cyphers:
            _execute_cyphers(cleanup_cyphers.splitlines(), db)
    return False


@pytest_mark.parametrize("test_dir", tests)
def test_end2end(test_dir: Path, db: Memgraph):
    db.drop_database()

    if test_dir.name.startswith(TestConstants.EXPORT_TEST_SUBDIR_PREFIX):
        _test_export(test_dir, db)
    elif test_dir.name.startswith(TestConstants.ONLINE_TEST_SUBDIR_PREFIX):
        _test_online(test_dir, db)
    else:
        _test_static(test_dir, db)

    # Clean database once testing module is finished
    db.drop_database()
    db.drop_indexes()
    db.ensure_constraints([])
    return False
