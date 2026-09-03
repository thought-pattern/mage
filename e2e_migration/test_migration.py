"""
Migration testing module for e2e_migration tests.
Tests the migration of data from various databases to Memgraph using the migrate module.
"""

from logging import INFO as logging_INFO
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger
from pathlib import Path

from gqlalchemy import Memgraph
from pytest import fail as pytest_fail
from pytest import fixture as pytest_fixture
from pytest import mark as pytest_mark
from pytest import param as pytest_param
from yaml import safe_load as yaml_safe_load


def load_test_config(test_file_path: str) -> dict[str, object]:
    """Load test configuration from YAML file."""
    try:
        with open(test_file_path, "r") as file:
            config = yaml_safe_load(file)
        if not isinstance(config, dict):
            raise TypeError("migration test configuration must be a dictionary")
        normalized_config: dict[str, object] = {}
        for key, value in config.items():
            if not isinstance(key, str):
                raise TypeError("migration test configuration keys must be strings")
            normalized_config[key] = value
        return normalized_config
    except Exception as e:
        logger.error(f"Failed to load test config from {test_file_path}: {e}")
        raise
    return {}


def prepare_migration_tests() -> list[object]:
    """
    Fetch all the migration tests and prepare them for execution.
    Returns a list of pytest.param objects with test directory and file information.
    """
    tests = []
    current_dir = Path(__file__).parent

    # Find all test_* directories
    for test_dir in current_dir.iterdir():
        if not test_dir.is_dir() or not test_dir.name.startswith("test_"):
            continue

        # Find all test files in the test subdirectory
        test_path = test_dir / "test"
        if not test_path.exists():
            continue

        for subdir in test_path.iterdir():
            if not subdir.is_dir():
                continue

            for yml_file in subdir.glob("*.yml"):
                # Create a descriptive test ID
                test_id = f"{test_dir.name}-{subdir.name}-{yml_file.stem}"

                tests.append(pytest_param(test_dir.name, str(yml_file.relative_to(test_dir)), id=test_id))

    computed_return_value = sorted(tests, key=lambda x: x.id)
    return computed_return_value


# Prepare all migration tests
migration_tests = prepare_migration_tests()


logging_basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging_getLogger("e2e_migration")
logger.setLevel(logging_INFO)


@pytest_fixture
def db():
    """Fixture to provide Memgraph database connection."""
    computed_return_value = Memgraph()
    return computed_return_value


@pytest_mark.parametrize("test_dir,test_file", migration_tests)
def test_migration(test_dir: str, test_file: str, db: Memgraph):
    # Load test configuration
    script_dir = Path(__file__).parent
    test_config_path = script_dir / test_dir / test_file
    test_config = load_test_config(str(test_config_path))

    # Check if this test expects an exception
    expect_exception = test_config.get("exception", False)

    try:
        # Execute migration query
        migration_query = test_config.get("query", "")
        if not isinstance(migration_query, str):
            raise TypeError("migration query must be a string")
        memgraph_results = list(db.execute_and_fetch(migration_query))

        if expect_exception:
            # If we expected an exception but got results, the test should fail
            pytest_fail("Expected migration to fail with an exception, but it succeeded")

        # Get expected results from test configuration
        expected_results = test_config.get("output", [])
        if not isinstance(expected_results, list):
            raise TypeError("migration output must be a list")

        # Compare results
        assert len(memgraph_results) == len(
            expected_results
        ), f"Result count mismatch: expected {len(expected_results)}, got {len(memgraph_results)}"

        # Validate data by comparing with expected output
        for i, (actual_row, expected_row) in enumerate(zip(memgraph_results, expected_results, strict=False)):
            if not isinstance(expected_row, dict):
                raise TypeError("each migration output row must be a dictionary")
            for field, expected_value in expected_row.items():
                actual_value = actual_row.get(field, False)

                # Simple equality comparison for all data types
                assert (
                    actual_value == expected_value
                ), f"Field {field} mismatch at row {i}: expected {expected_value}, got {actual_value}"

        logger.info("Data migration to Memgraph successfully validated!")

    except Exception as e:
        if expect_exception:
            # Expected exception - test passes
            logger.info(f"Migration failed as expected: {e}")
            return False
        else:
            # Unexpected failure - re-raise the exception
            raise
