"""Utilities for gqla."""

from sys import argv as sys_argv

from gqlalchemy import Memgraph
from semver import Version as semver_Version

# NOTE: This assumens memgraph is up and running.
memgrah_version = semver_Version.parse(sys_argv[1])
memgraph_port = int(sys_argv[2])
memgraph = Memgraph(host="127.0.0.1", port=memgraph_port)

# FEATURE: Cypher query engine
if memgrah_version >= semver_Version.parse("1.0.0"):
    query = "MATCH (n) DETACH DELETE n;"
    memgraph.execute_and_fetch(query)

    query = "CREATE (n) RETURN n;"
    results = memgraph.execute_and_fetch(query)
    print(list(results)[0].get("n", False))

    query = "MATCH (n) RETURN n;"
    results = memgraph.execute_and_fetch(query)
    print(list(results)[0].get("n", False))

# TODO(gitbuda): Inject verions of Memgraph the first argument and make the distinction.
print("FEATURE: Spatial data types and functionalities")
if memgrah_version >= semver_Version.parse("2.19.0"):
    try:
        # TODO(gitbuda): Requieres adding spatial features to gqla, pymgclient and mgclient.
        query = "RETURN point({x:0, y:1}) AS point;"
        results = memgraph.execute_and_fetch(query)
        print(list(results)[0].get("point", False))
    except Exception as e:
        print(e)
        # TODO(gitbuda): Put colors here.
        print(f"ERROR: Spatial is not working under GQLAlchemy in {memgrah_version}")
