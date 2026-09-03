"""Utilities for gqla."""

from sys import argv as sys_argv

from gqlalchemy import Memgraph

# This probe requires a running current Memgraph instance.
memgraph_port = int(sys_argv[1])
memgraph = Memgraph(host="127.0.0.1", port=memgraph_port)

query = "MATCH (n) DETACH DELETE n;"
memgraph.execute_and_fetch(query)

query = "CREATE (n) RETURN n;"
results = memgraph.execute_and_fetch(query)
print(list(results)[0].get("n", False))

query = "MATCH (n) RETURN n;"
results = memgraph.execute_and_fetch(query)
print(list(results)[0].get("n", False))

print("FEATURE: Spatial data types and functionalities")
query = "RETURN point({x:0, y:1}) AS point;"
results = memgraph.execute_and_fetch(query)
print(list(results)[0].get("point", False))
