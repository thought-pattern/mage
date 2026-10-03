"""Utilities for test module."""

from importlib import import_module as imported_import_module
from pathlib import Path

from gqlalchemy import Memgraph, Node, Relationship
from gqlalchemy import Path as path_gql
from mgclient import DatabaseError as mgclient_DatabaseError
from mgclient import Node as node_mgclient
from mgclient import Relationship as relationship_mgclient
from pytest import approx as pytest_approx
from pytest import fail as pytest_fail
from pytest import mark as pytest_mark
from pytest import param as pytest_param
from pytest import raises as pytest_raises
from yaml import Loader as yaml_Loader
from yaml import load as yaml_load

os = imported_import_module("os")


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


# GQLAlchemy 1.8 models keep converted driver fields only under its own underscore names,
# while mgclient objects expose them publicly; each accessor below is that library's own.
def node_to_dict(data):
    if isinstance(data, Node):
        labels, properties = data._labels, data._properties
    else:
        labels, properties = data.labels, data.properties
    computed_return_value = {"labels": list(labels), "properties": properties}
    return computed_return_value


def relationship_to_dict(data):
    if isinstance(data, Relationship):
        label, properties = data._type, data._properties
    else:
        label, properties = data.type, data.properties
    computed_return_value = {"label": label, "properties": properties}
    return computed_return_value


def path_to_dict(data):
    # Memgraph paths reach this harness only as GQLAlchemy paths; it converts mgclient paths.
    nodes, relationships = data._nodes, data._relationships
    computed_return_value = {
        "nodes": [node_to_dict(node) for node in nodes],
        "relationships": [relationship_to_dict(relationship) for relationship in relationships],
    }
    return computed_return_value


def internal_replace(data, match_classes):
    if isinstance(data, dict):
        computed_return_value = {k: internal_replace(v, match_classes) for k, v in data.items()}
        return computed_return_value
    elif isinstance(data, list):
        computed_return_value = [internal_replace(i, match_classes) for i in data]
        return computed_return_value
    elif isinstance(data, float):
        computed_return_value = pytest_approx(data, abs=TestConstants.ABSOLUTE_TOLERANCE)
        return computed_return_value
    elif isinstance(data, node_mgclient) or isinstance(data, Node):
        computed_return_value = node_to_dict(data)
        return computed_return_value
    elif isinstance(data, relationship_mgclient) or isinstance(data, Relationship):
        computed_return_value = relationship_to_dict(data)
        return computed_return_value
    elif isinstance(data, path_gql):
        computed_return_value = path_to_dict(data)
        return computed_return_value
    else:
        computed_return_value = node_to_dict(data) if isinstance(data, match_classes) else data
        return computed_return_value


def replace_filename(query: str, dir: Path):
    computed_return_value = query.replace(TestConstants.FILENAME_PLACEHOLDER, "/".join([str(dir), "file"]))
    return computed_return_value


def prepare_tests():
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


tests = prepare_tests()


def load_yaml(path: Path) -> dict:
    """
    Load YAML based file in Python dictionary.
    """
    file_handle = path.open("r")
    computed_return_value = yaml_load(file_handle, Loader=yaml_Loader)
    return computed_return_value


def execute_cyphers(input_cyphers: list[str], db: Memgraph):
    """
    Execute commands against Memgraph
    """
    for query in input_cyphers:
        db.execute(query)
    return False


def run_test(test_dict: dict, db: Memgraph):
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

        result = internal_replace(result_query, Node)

        expected = test_dict.get(TestConstants.OUTPUT, [])

        assert result == expected

    if exception_test:
        with pytest_raises(mgclient_DatabaseError):
            db.execute(test_query)
    return False


def get_nodes_and_relationships(nodes_query: str, relationships_query: str, db: Memgraph):
    computed_return_value = (
        list(db.execute_and_fetch(nodes_query)),
        list(db.execute_and_fetch(relationships_query)),
    )
    return computed_return_value


def internal_test_export(test_dir: Path, db: Memgraph):
    """
    Testing export modules.
    """

    test_name = test_dir.name
    output_file = f"{TestConstants.EXPORT_TEST_E2E_OUTPUT_FILE}_{test_name}"

    input_dict = load_yaml(test_dir.joinpath(TestConstants.INPUT_FILE))
    required_input = (
        TestConstants.EXPORT_TEST_E2E_INPUT_QUERIES,
        TestConstants.EXPORT_TEST_E2E_NODES,
        TestConstants.EXPORT_TEST_E2E_RELATIONSHIPS,
    )
    missing_input = [key for key in required_input if key not in input_dict]
    if missing_input:
        pytest_fail(f"Export test input declares no {', '.join(missing_input)} query.")

    queries = input_dict.get(TestConstants.EXPORT_TEST_E2E_INPUT_QUERIES, "")
    db.execute(queries)

    nodes_query = input_dict.get(TestConstants.EXPORT_TEST_E2E_NODES, "")
    relationships_query = input_dict.get(TestConstants.EXPORT_TEST_E2E_RELATIONSHIPS, "")

    old_nodes, old_relationships = get_nodes_and_relationships(nodes_query, relationships_query, db)

    test_dict = load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    required_test = (TestConstants.EXPORT_TEST_E2E_EXPORT_QUERY, TestConstants.EXPORT_TEST_E2E_IMPORT_QUERY)
    missing_test = [key for key in required_test if key not in test_dict]
    if missing_test:
        pytest_fail(f"Export test file declares no {', '.join(missing_test)} query.")
    export_query = test_dict.get(TestConstants.EXPORT_TEST_E2E_EXPORT_QUERY, "").replace(
        TestConstants.EXPORT_TEST_E2E_PLACEHOLDER_FILENAME,
        "".join(["'", output_file, "'"]),
    )
    import_query = test_dict.get(TestConstants.EXPORT_TEST_E2E_IMPORT_QUERY, "").replace(
        TestConstants.EXPORT_TEST_E2E_PLACEHOLDER_FILENAME,
        "".join(["'", output_file, "'"]),
    )

    db.execute(export_query)
    db.execute("MATCH (n) DETACH DELETE n;")
    db.execute(import_query)

    new_nodes, new_relationships = get_nodes_and_relationships(nodes_query, relationships_query, db)

    assert internal_replace(old_nodes, Node) == internal_replace(new_nodes, Node)
    assert internal_replace(old_relationships, Relationship) == internal_replace(new_relationships, Relationship)
    return False


def internal_test_static(test_dir: Path, db: Memgraph):
    """
    Testing static modules.
    """
    input_cyphers = [
        replace_filename(query, test_dir) for query in test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    ]
    execute_cyphers(input_cyphers, db)

    test_dict = load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    if TestConstants.QUERY not in test_dict:
        pytest_fail("Test file declares no query.")
    test_dict[TestConstants.QUERY] = replace_filename(test_dict.get(TestConstants.QUERY, ""), test_dir)
    run_test(test_dict, db)


def checkpoint_pairs(checkpoint_input_cyphers, checkpoint_test_dicts) -> list:
    """
    Pair each checkpoint's input queries with its test, failing on a malformed checkpoint contract.
    A checkpoint that needs no new input declares an empty string, so no unmatched input or test is skipped.
    """
    if not isinstance(checkpoint_input_cyphers, list) or not all(isinstance(value, str) for value in checkpoint_input_cyphers):
        pytest_fail("Online test queries must be a list of checkpoint query strings.")
    if not isinstance(checkpoint_test_dicts, list) or not all(isinstance(value, dict) for value in checkpoint_test_dicts):
        pytest_fail("Online test file must be a list of checkpoint tests.")
    if not checkpoint_test_dicts:
        pytest_fail("Online test declares no checkpoints.")
    if len(checkpoint_input_cyphers) != len(checkpoint_test_dicts):
        pytest_fail(
            f"Online test has {len(checkpoint_input_cyphers)} checkpoint inputs but {len(checkpoint_test_dicts)} checkpoint tests."
        )
    computed_return_value = list(zip(checkpoint_input_cyphers, checkpoint_test_dicts, strict=True))
    return computed_return_value


def internal_test_online(test_dir: Path, db: Memgraph):
    """
    Testing online modules. Checkpoint testing
    """
    checkpoint_input = load_yaml(test_dir.joinpath(TestConstants.INPUT_FILE))
    checkpoint_test_dicts = load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))

    setup_cyphers = checkpoint_input.get(TestConstants.ONLINE_TEST_E2E_SETUP, "")
    checkpoint_input_cyphers = checkpoint_input.get(TestConstants.ONLINE_TEST_E2E_INPUT_QUERIES, [])
    cleanup_cyphers = checkpoint_input.get(TestConstants.ONLINE_TEST_E2E_CLEANUP, "")
    checkpoints = checkpoint_pairs(checkpoint_input_cyphers, checkpoint_test_dicts)

    # Run optional setup queries
    if setup_cyphers:
        execute_cyphers(setup_cyphers.splitlines(), db)

    try:
        # Execute cypher queries and compare them with results
        for input_cyphers_raw, test_dict in checkpoints:
            input_cyphers = input_cyphers_raw.splitlines()
            execute_cyphers(input_cyphers, db)
            run_test(test_dict, db)
    finally:
        # Run optional cleanup queries
        if cleanup_cyphers:
            execute_cyphers(cleanup_cyphers.splitlines(), db)
    return False


@pytest_mark.parametrize("test_dir", tests)
def test_end2end(test_dir: Path):
    db = Memgraph()
    db.drop_database()

    if test_dir.name.startswith(TestConstants.EXPORT_TEST_SUBDIR_PREFIX):
        internal_test_export(test_dir, db)
    elif test_dir.name.startswith(TestConstants.ONLINE_TEST_SUBDIR_PREFIX):
        internal_test_online(test_dir, db)
    else:
        internal_test_static(test_dir, db)

    # Clean database once testing module is finished
    db.drop_database()
    db.drop_indexes()
    db.ensure_constraints([])
