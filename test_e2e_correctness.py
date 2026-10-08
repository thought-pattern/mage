"""Utilities for test e2e correctness."""

from argparse import ArgumentParser as argparse_ArgumentParser
from os import chdir as os_chdir, environ as os_environ, getcwd as os_getcwd
from subprocess import CalledProcessError as subprocess_CalledProcessError, run as subprocess_run
from sys import exit as sys_exit

WORK_DIRECTORY = os_getcwd()
E2E_CORRECTNESS_DIRECTORY = f"{WORK_DIRECTORY}/e2e_correctness"


class ConfigConstants:
    NEO4J_PORT = 7688
    MEMGRAPH_PORT = 7687
    NEO4J_CONTAINER_NAME = "neo4j"


def parse_arguments():
    parser = argparse_ArgumentParser(description="Test MAGE E2E correctness.")
    parser.add_argument(
        "-k",
        help="Filter what tests you want to run",
        type=str,
        default="",
    )
    parser.add_argument(
        "--memgraph-port",
        help="Set the port that Memgraph is listening on",
        type=int,
        default=ConfigConstants.MEMGRAPH_PORT,
    )
    parser.add_argument(
        "--neo4j-port",
        help="Set the port that Neo4j is listening on",
        type=int,
        default=ConfigConstants.NEO4J_PORT,
    )
    parser.add_argument(
        "--neo4j-container",
        help="Set the Neo4j container name",
        type=str,
        default=ConfigConstants.NEO4J_CONTAINER_NAME,
    )
    args = parser.parse_args()
    return args


#################################################
#                End to end tests               #
#################################################


def main(
    test_filter: str = "",
    memgraph_port: str = str(ConfigConstants.MEMGRAPH_PORT),
    neo4j_port: str = str(ConfigConstants.NEO4J_PORT),
    neo4j_container: str = ConfigConstants.NEO4J_CONTAINER_NAME,
):
    os_environ["PYTHONPATH"] = E2E_CORRECTNESS_DIRECTORY
    os_chdir(E2E_CORRECTNESS_DIRECTORY)
    command = ["python3", "-m", "pytest", ".", "-vv"]
    if test_filter:
        command.extend(["-k", test_filter])

    command.extend(["--memgraph-port", memgraph_port])
    command.extend(["--neo4j-port", neo4j_port])
    command.extend(["--neo4j-container", neo4j_container])

    try:
        subprocess_run(command, check=True)
    except subprocess_CalledProcessError as err:
        print(f"Error: {err}")
        sys_exit(err.returncode)
    return False


if __name__ == "__main__":
    args = parse_arguments()

    main(
        test_filter=args.k,
        memgraph_port=str(args.memgraph_port),
        neo4j_port=str(args.neo4j_port),
        neo4j_container=args.neo4j_container,
    )
