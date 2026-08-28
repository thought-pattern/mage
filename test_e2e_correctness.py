#!/usr/bin/env python3

"""Utilities for test e2e correctness."""

from argparse import ArgumentParser as argparse_ArgumentParser
from os import chdir as os_chdir
from os import environ as os_environ
from os import getcwd as os_getcwd
from subprocess import CalledProcessError as subprocess_CalledProcessError
from subprocess import run as subprocess_run
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
        required=False,
    )
    parser.add_argument(
        "--memgraph-port",
        help="Set the port that Memgraph is listening on",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--neo4j-port",
        help="Set the port that Neo4j is listening on",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--neo4j-container",
        help="Set the Neo4j container name",
        type=str,
        required=False,
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
    except subprocess_CalledProcessError as e:
        print(f"Error: {e}")
        sys_exit(e.returncode)
    return False


if __name__ == "__main__":
    args = parse_arguments()
    test_filter = args.k
    memgraph_port = args.memgraph_port
    neo4j_port = args.neo4j_port
    neo4j_container = args.neo4j_container

    if memgraph_port:
        memgraph_port = str(memgraph_port)
    if neo4j_port:
        neo4j_port = str(neo4j_port)
    if args.neo4j_container:
        neo4j_container = args.neo4j_container

    main(
        test_filter=test_filter,
        memgraph_port=memgraph_port,
        neo4j_port=neo4j_port,
        neo4j_container=neo4j_container,
    )
