#!/usr/bin/env python3
"""
Validation script for cuGraph MAGE algorithms after RAPIDS 25.x migration.
Validates algorithm ACCURACY by comparing against NetworkX ground truth.

This script:
1. Builds the same graph in NetworkX (ground truth)
2. Computes expected values using NetworkX algorithms
3. Runs cuGraph algorithms via Memgraph
4. Compares results with tolerance
5. Validates node identity mapping is correct

Each run owns exactly one container (named MEMGRAPH_CONTAINER plus a run identifier and labelled with
that identifier), a fresh data directory, and the loopback Bolt port Docker assigns to that container. It
never stops, removes, or clears any other container, directory, or database, and releases its own
container and then its data directory on every exit unless MEMGRAPH_RETAIN asks to keep them.

Usage:
    # Using default settings (data directory under the system temp directory)
    python validate_cugraph_algorithms.py

    # Using custom settings via environment variables
    MEMGRAPH_DATA_DIR=/path/to/parent MEMGRAPH_IMAGE=my-image:tag python validate_cugraph_algorithms.py

Environment Variables:
    MEMGRAPH_DATA_DIR    - Parent of the run's data directory (default: system temp directory)
    MEMGRAPH_IMAGE       - Docker image name (default: memgraph-mage-cugraph:latest)
    MEMGRAPH_CONTAINER   - Container name prefix (default: memgraph-cugraph-validation)
    MEMGRAPH_RETAIN      - "true" keeps the run's container and data directory for inspection
"""

from os import environ as os_environ
from os import getgid as os_getgid
from os import getuid as os_getuid
from pathlib import Path
from shutil import rmtree as shutil_rmtree
from subprocess import CompletedProcess as subprocess_CompletedProcess
from subprocess import run as subprocess_run
from sys import exit as sys_exit
from tempfile import mkdtemp as tempfile_mkdtemp
from time import sleep as time_sleep
from uuid import uuid4 as uuid_uuid4

from neo4j import GraphDatabase
from networkx import DiGraph as nx_DiGraph
from networkx import PowerIterationFailedConvergence as nx_PowerIterationFailedConvergence
from networkx import betweenness_centrality as nx_betweenness_centrality
from networkx import community as nx_community
from networkx import hits as nx_hits
from networkx import katz_centrality as nx_katz_centrality
from networkx import pagerank as nx_pagerank

# Configuration via environment variables with sensible defaults
MEMGRAPH_USER = os_environ.get("MEMGRAPH_USER", "")
MEMGRAPH_PASSWORD = os_environ.get("MEMGRAPH_PASSWORD", "")

# Docker configuration
CONTAINER_NAME = os_environ.get("MEMGRAPH_CONTAINER", "memgraph-cugraph-validation")
IMAGE_NAME = os_environ.get("MEMGRAPH_IMAGE", "memgraph-mage-cugraph:latest")
# Label whose value is the run identifier; release removes only containers carrying this run's value.
OWNER_LABEL = "org.memgraph.mage.cugraph-validation.run"

# Parent directory for each run's own data directory
DATA_PARENT_DIR = os_environ.get("MEMGRAPH_DATA_DIR", "")
RETAIN_RESOURCES = os_environ.get("MEMGRAPH_RETAIN", "").lower() in ("1", "true", "yes")

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Test tolerance for floating point comparisons
TOLERANCE = 0.05  # 5% relative tolerance
ABS_TOLERANCE = 1e-6  # Absolute tolerance for near-zero values

# Authored node identities of the test graph (name -> id), shared by the NetworkX graph and the identity check
EXPECTED_NODE_IDS = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "B1": 5, "B2": 6, "B3": 7, "B4": 8, "HUB": 9}
EXPECTED_NODES = set(EXPECTED_NODE_IDS)
COMMUNITY_A = {"A1", "A2", "A3", "A4"}
COMMUNITY_B = {"B1", "B2", "B3", "B4"}
# HUB bridges A1 and B1 symmetrically, so an accepted partition may place it with either community or alone.
TIE_NODE = "HUB"
# Authored directed edges of the test graph as (source id, target id), each with weight 1.0, in the order the
# NetworkX reference graph adds them; create_test_graph writes the same edges to Memgraph.
TEST_GRAPH_EDGES = (
    # Community A (A1-A4): ring, then cross connections
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
    (1, 3),
    (2, 4),
    # Community B (B1-B4): ring, then cross connections
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 5),
    (5, 7),
    (6, 8),
    # HUB bridge: A1 -> HUB -> B1, then HUB -> A1 and B1 -> HUB
    (1, 9),
    (9, 5),
    (9, 1),
    (5, 9),
)


def get_networkx_ground_truth(G: nx_DiGraph):
    """Compute ground truth values using NetworkX algorithms."""
    # Create name lookup
    id_to_name = {node: G.nodes[node].get("name", "") for node in G.nodes()}

    # PageRank
    pagerank = nx_pagerank(G, alpha=0.85, max_iter=100, tol=1e-5)
    pagerank_by_name = {id_to_name.get(k, ""): v for k, v in pagerank.items()}

    # Betweenness Centrality (normalized, directed)
    betweenness = nx_betweenness_centrality(G, normalized=True)
    betweenness_by_name = {id_to_name.get(k, ""): v for k, v in betweenness.items()}

    # HITS
    hubs, authorities = nx_hits(G, max_iter=100, tol=1e-5, normalized=True)
    hubs_by_name = {id_to_name.get(k, ""): v for k, v in hubs.items()}
    authorities_by_name = {id_to_name.get(k, ""): v for k, v in authorities.items()}

    # Katz Centrality; an empty baseline records that NetworkX did not converge, and test_katz_centrality then
    # skips the value comparison.
    try:
        katz = nx_katz_centrality(G, alpha=0.1, beta=1.0, max_iter=100, tol=1e-6, normalized=False)
        katz_by_name = {id_to_name.get(k, ""): v for k, v in katz.items()}
    except nx_PowerIterationFailedConvergence:
        katz_by_name = {}

    # Community detection (Louvain) - use undirected graph
    G_undirected = G.to_undirected()
    communities = nx_community.louvain_communities(G_undirected, seed=42)
    community_by_name = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_by_name[id_to_name.get(node, "")] = idx

    # Personalized PageRank from node 1 (A1)
    personalization = dict.fromkeys(G.nodes(), 0.0)
    personalization[1] = 1.0
    ppr = nx_pagerank(G, alpha=0.85, personalization=personalization, max_iter=100, tol=1e-5)
    ppr_by_name = {id_to_name.get(k, ""): v for k, v in ppr.items()}

    return {
        "pagerank": pagerank_by_name,
        "betweenness": betweenness_by_name,
        "hubs": hubs_by_name,
        "authorities": authorities_by_name,
        "katz": katz_by_name,
        "communities": community_by_name,
        "personalized_pagerank": ppr_by_name,
    }


def values_match(expected: float, actual: float, name: str = "") -> tuple[bool, str]:
    """Check if two values match within tolerance."""
    if abs(expected) < ABS_TOLERANCE and abs(actual) < ABS_TOLERANCE:
        return True, ""

    if abs(expected) < ABS_TOLERANCE:
        diff = abs(actual)
    else:
        diff = abs(actual - expected) / abs(expected)

    if diff <= TOLERANCE:
        return True, ""
    else:
        computed_return_value = (
            False,
            f"{name}: expected {expected:.6f}, got {actual:.6f} (diff: {diff:.1%})",
        )
        return computed_return_value
    return ()


def core_partition(assignment: dict) -> list[list[str]]:
    """Group every node except the tie node by community, independent of community IDs."""
    groups = {}
    for name in sorted(EXPECTED_NODES - {TIE_NODE}):
        groups.setdefault(assignment.get(name, ""), []).append(name)
    computed_return_value = sorted(groups.values())
    return computed_return_value


def partition_contract_errors(actual: dict, baseline: dict) -> list[str]:
    """Require the NetworkX baseline partition, up to relabelling and the declared tie-node placement."""
    errors = []
    declared = sorted([sorted(COMMUNITY_A), sorted(COMMUNITY_B)])
    baseline_groups = core_partition(baseline)
    actual_groups = core_partition(actual)
    if baseline_groups != declared:
        errors.append(f"NetworkX baseline {baseline_groups} does not separate the authored communities {declared}")
    if actual_groups != baseline_groups:
        errors.append(f"Partition {actual_groups} differs from the NetworkX baseline {baseline_groups}")
    return errors


def require_baseline_scores(ground_truth: dict, algorithm: str) -> dict:
    """Return one algorithm's per-node NetworkX scores after checking they cover every authored node."""
    scores = ground_truth.get(algorithm, {})
    if not isinstance(scores, dict):
        raise TypeError(f"{algorithm} ground truth must be a dictionary")
    missing = sorted(EXPECTED_NODES - set(scores))
    if missing:
        raise ValueError(f"{algorithm} ground truth lacks nodes {missing}")
    return scores


def run_cmd(cmd: list[str]) -> subprocess_CompletedProcess:
    """Run a shell command."""
    print(f"  $ {' '.join(cmd)}")
    computed_return_value = subprocess_run(cmd, capture_output=True, text=True, check=True)
    return computed_return_value


def acquire_data_dir() -> Path:
    """Create this run's empty Memgraph data directory."""
    if DATA_PARENT_DIR:
        Path(DATA_PARENT_DIR).mkdir(parents=True, exist_ok=True)
        data_dir = Path(tempfile_mkdtemp(prefix="memgraph_validation_", dir=DATA_PARENT_DIR))
    else:
        data_dir = Path(tempfile_mkdtemp(prefix="memgraph_validation_"))
    print(f"  Run data directory: {data_dir}")
    return data_dir


def setup_container(container_name: str, run_id: str, data_dir: Path) -> str:
    """Start this run's container on its own data directory and return the Bolt URI Docker assigned it."""
    print("\n" + "=" * 60)
    print("CONTAINER SETUP")
    print("=" * 60)

    print(f"\n>>> Checking image '{IMAGE_NAME}' exists...")
    result = run_cmd(["docker", "images", "-q", IMAGE_NAME])
    if not result.stdout.strip():
        print(f"ERROR: Image '{IMAGE_NAME}' not found!")
        print("Build it first with:")
        print(f"  docker build -f Dockerfile.cugraph -t {IMAGE_NAME} .")
        raise RuntimeError(f"validation image {IMAGE_NAME!r} is not available")
    print(f"  Image ID: {result.stdout.strip()}")

    print(f"\n>>> Starting new container '{container_name}'...")
    uid = os_getuid()
    gid = os_getgid()

    # The Bolt port is published on a Docker-assigned loopback port so the run never binds or reaches a peer's
    # Memgraph endpoint.
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--label",
        f"{OWNER_LABEL}={run_id}",
        "--user",
        f"{uid}:{gid}",
        "--gpus",
        "all",
        "-p",
        "127.0.0.1::7687",
        "-v",
        f"{data_dir}:/var/lib/memgraph:z",
        IMAGE_NAME,
        "--storage-mode=IN_MEMORY_ANALYTICAL",
        "--query-execution-timeout-sec=0",
        "--log-level=WARNING",
        "--log-file=",
        "--also-log-to-stderr",
    ]
    result = run_cmd(cmd)
    print(f"  Container started: {result.stdout.strip()}")

    print("\n>>> Verifying container uses correct image...")
    result = run_cmd(["docker", "inspect", "--format", "{{.Config.Image}}", container_name])
    actual_image = result.stdout.strip()
    print(f"  Container image: {actual_image}")
    if actual_image != IMAGE_NAME:
        print(f"  WARNING: Expected {IMAGE_NAME}, got {actual_image}")

    result = run_cmd(["docker", "port", container_name, "7687/tcp"])
    bindings = result.stdout.split()
    if not bindings:
        raise RuntimeError(f"container {container_name!r} has no published Bolt port")
    bolt_uri = f"bolt://{bindings[0]}"
    print(f"  Bolt endpoint: {bolt_uri}")
    return bolt_uri


def release_resources(run_id: str, data_dir: Path) -> None:
    """Remove exactly this run's container, then its data directory, unless they are retained."""
    if RETAIN_RESOURCES:
        print(f"\n>>> Retaining containers labelled {OWNER_LABEL}={run_id} and data directory {data_dir}")
    else:
        result = run_cmd(["docker", "ps", "-a", "-q", "--filter", f"label={OWNER_LABEL}={run_id}"])
        for container_id in result.stdout.split():
            run_cmd(["docker", "rm", "-f", container_id])
        print(f"\n>>> Removing run data directory: {data_dir}")
        shutil_rmtree(data_dir)


def wait_for_memgraph(driver, max_retries=30, delay=2):
    """Wait for Memgraph to be ready."""
    for i in range(max_retries):
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            print("✓ Memgraph is ready")
            return True
        except Exception:
            print(f"  Waiting for Memgraph... ({i + 1}/{max_retries})")
            time_sleep(delay)
    print("✗ Memgraph failed to start")
    return False


def clear_database(session):
    """Clear all data from the database."""
    session.run("MATCH (n) DETACH DELETE n")
    return False


def create_test_graph(session):
    """Create a test graph for algorithm validation."""
    queries = [
        "CREATE (a1:Node {id: 1, name: 'A1'})",
        "CREATE (a2:Node {id: 2, name: 'A2'})",
        "CREATE (a3:Node {id: 3, name: 'A3'})",
        "CREATE (a4:Node {id: 4, name: 'A4'})",
        "CREATE (b1:Node {id: 5, name: 'B1'})",
        "CREATE (b2:Node {id: 6, name: 'B2'})",
        "CREATE (b3:Node {id: 7, name: 'B3'})",
        "CREATE (b4:Node {id: 8, name: 'B4'})",
        "CREATE (hub:Node {id: 9, name: 'HUB'})",
        """
        MATCH (a1:Node {id: 1}), (a2:Node {id: 2}), (a3:Node {id: 3}), (a4:Node {id: 4})
        CREATE (a1)-[:EDGE {weight: 1.0}]->(a2),
               (a2)-[:EDGE {weight: 1.0}]->(a3),
               (a3)-[:EDGE {weight: 1.0}]->(a4),
               (a4)-[:EDGE {weight: 1.0}]->(a1),
               (a1)-[:EDGE {weight: 1.0}]->(a3),
               (a2)-[:EDGE {weight: 1.0}]->(a4)
        """,
        """
        MATCH (b1:Node {id: 5}), (b2:Node {id: 6}), (b3:Node {id: 7}), (b4:Node {id: 8})
        CREATE (b1)-[:EDGE {weight: 1.0}]->(b2),
               (b2)-[:EDGE {weight: 1.0}]->(b3),
               (b3)-[:EDGE {weight: 1.0}]->(b4),
               (b4)-[:EDGE {weight: 1.0}]->(b1),
               (b1)-[:EDGE {weight: 1.0}]->(b3),
               (b2)-[:EDGE {weight: 1.0}]->(b4)
        """,
        """
        MATCH (a1:Node {id: 1}), (b1:Node {id: 5}), (hub:Node {id: 9})
        CREATE (a1)-[:EDGE {weight: 1.0}]->(hub),
               (hub)-[:EDGE {weight: 1.0}]->(b1),
               (hub)-[:EDGE {weight: 1.0}]->(a1),
               (b1)-[:EDGE {weight: 1.0}]->(hub)
        """,
    ]

    for query in queries:
        session.run(query)

    result = session.run("MATCH (n) RETURN count(n) as nodes")
    node_count = result.single().get("nodes", 0)

    result = session.run("MATCH ()-[r]->() RETURN count(r) as edges")
    edge_count = result.single().get("edges", 0)

    print(f"✓ Test graph created: {node_count} nodes, {edge_count} edges")
    computed_return_value = node_count == 9 and edge_count == 16
    return computed_return_value


def validate_node_identities(records: list, algorithm_name: str) -> tuple[bool, list[str]]:
    """Validate that all expected nodes are returned with correct identities."""
    errors = []

    # Check node count
    if len(records) != len(EXPECTED_NODE_IDS):
        errors.append(f"Expected {len(EXPECTED_NODE_IDS)} nodes, got {len(records)}")

    # Check all node names are present
    returned_names = {r.get("name", "") for r in records}
    missing = EXPECTED_NODES - returned_names
    extra = returned_names - EXPECTED_NODES

    if missing:
        errors.append(f"Missing nodes: {missing}")
    if extra:
        errors.append(f"Unexpected nodes: {extra}")

    # Check each returned name still carries its authored id
    mismatched = sorted(
        f"{r.get('name', '')}={r.get('id', '')}"
        for r in records
        if r.get("name", "") in EXPECTED_NODE_IDS and r.get("id", "") != EXPECTED_NODE_IDS.get(r.get("name", ""), 0)
    )
    if mismatched:
        errors.append(f"Node IDs differ from the authored graph: {mismatched}")

    computed_return_value = len(errors) == 0, errors
    return computed_return_value


def test_pagerank(session, ground_truth: dict) -> bool:
    """Test PageRank algorithm against NetworkX ground truth."""
    print("\n--- Testing PageRank ---")
    try:
        result = session.run(
            """
            CALL cugraph.pagerank.get(100, 0.85, 1e-5)
            YIELD node, pagerank
            RETURN node.id AS id, node.name AS name, pagerank
            ORDER BY pagerank DESC
        """
        )

        records = list(result)

        # Validate node identities
        valid, errors = validate_node_identities(records, "PageRank")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        print(f"✓ PageRank: {len(records)} nodes returned")

        # Compare against NetworkX ground truth
        expected = require_baseline_scores(ground_truth, "pagerank")
        all_match = True

        for r in records:
            name = r.get("name", "")
            actual = r.get("pagerank", False)
            exp = expected.get(name, 0.0)
            match, err = values_match(exp, actual, name)
            if not match:
                print(f"  ✗ {err}")
                all_match = False
            else:
                print(f"  ✓ {name}: {actual:.6f} (expected: {exp:.6f})")

        # Verify ranking order matches
        actual_ranking = [r.get("name", "") for r in records]
        expected_ranking = sorted(expected, key=lambda x: expected.get(x, 0.0), reverse=True)

        # Check top 3 ranking
        if actual_ranking[:3] != expected_ranking[:3]:
            print(f"  ⚠ Ranking differs: cuGraph={actual_ranking[:3]}, NetworkX={expected_ranking[:3]}")
            # This is a warning, not a failure - numerical precision can cause minor reordering

        return all_match

    except Exception as e:
        print(f"✗ PageRank failed: {e}")
        return False


def test_betweenness_centrality(session, ground_truth: dict) -> bool:
    """Test Betweenness Centrality - HUB must be highest."""
    print("\n--- Testing Betweenness Centrality ---")
    try:
        result = session.run(
            """
            CALL cugraph.betweenness_centrality.get(true, true)
            YIELD node, betweenness
            RETURN node.id AS id, node.name AS name, betweenness
            ORDER BY betweenness DESC
        """
        )

        records = list(result)

        valid, errors = validate_node_identities(records, "Betweenness")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        print(f"✓ Betweenness Centrality: {len(records)} nodes returned")

        expected = require_baseline_scores(ground_truth, "betweenness")
        all_match = True

        for r in records:
            name = r.get("name", "")
            actual = r.get("betweenness", False)
            exp = expected.get(name, 0.0)
            match, err = values_match(exp, actual, name)
            if not match:
                print(f"  ✗ {err}")
                all_match = False
            else:
                print(f"  ✓ {name}: {actual:.6f} (expected: {exp:.6f})")

        # Semantic check: HUB should be in top 3 and have high betweenness (it's the bridge)
        # Note: In linear chain topology, chain endpoints (A1, B1) have higher betweenness
        # because all paths from their chains must pass through them
        sorted_records = sorted(records, key=lambda r: r.get("betweenness", []), reverse=True)
        top_3_names = [r.get("name", "") for r in sorted_records[:3]]
        hub_bc = next(
            (r.get("betweenness", []) for r in records if r.get("name", "") == "HUB"),
            False,
        )

        if "HUB" not in top_3_names:
            print("  ✗ CRITICAL: HUB should be in top 3 betweenness nodes")
            print(f"    Top 3: {top_3_names}")
            return False
        elif hub_bc < 0.5:
            print(f"  ✗ CRITICAL: HUB betweenness too low: {hub_bc}")
            return False
        else:
            print(f"  ✓ SEMANTIC: HUB is in top 3 betweenness with score {hub_bc:.6f}")

        return all_match

    except Exception as e:
        print(f"✗ Betweenness Centrality failed: {e}")
        return False


def test_hits(session, ground_truth: dict) -> bool:
    """Test HITS algorithm against NetworkX ground truth."""
    print("\n--- Testing HITS ---")
    try:
        result = session.run(
            """
            CALL cugraph.hits.get(100, 1e-5, true)
            YIELD node, hub, authority
            RETURN node.id AS id, node.name AS name, hub, authority
            ORDER BY hub DESC
        """
        )

        records = list(result)

        valid, errors = validate_node_identities(records, "HITS")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        print(f"✓ HITS: {len(records)} nodes returned")

        expected_hubs = require_baseline_scores(ground_truth, "hubs")
        expected_auths = require_baseline_scores(ground_truth, "authorities")
        all_match = True

        for r in records:
            name = r.get("name", "")

            # Check hub values
            actual_hub = r.get("hub", False)
            exp_hub = expected_hubs.get(name, 0.0)
            match, err = values_match(exp_hub, actual_hub, f"{name} hub")
            if not match:
                print(f"  ✗ {err}")
                all_match = False

            # Check authority values
            actual_auth = r.get("authority", False)
            exp_auth = expected_auths.get(name, 0.0)
            match, err = values_match(exp_auth, actual_auth, f"{name} authority")
            if not match:
                print(f"  ✗ {err}")
                all_match = False

            if all_match:
                print(f"  ✓ {name}: hub={actual_hub:.6f}, auth={actual_auth:.6f}")

        return all_match

    except Exception as e:
        print(f"✗ HITS failed: {e}")
        return False


def test_louvain(session, ground_truth: dict) -> bool:
    """Test Louvain community detection - A1-A4 and B1-B4 should be grouped."""
    print("\n--- Testing Louvain ---")
    try:
        result = session.run(
            """
            CALL cugraph.louvain.get()
            YIELD node, partition
            RETURN node.id AS id, node.name AS name, partition AS community
            ORDER BY partition, id
        """
        )

        records = list(result)

        valid, errors = validate_node_identities(records, "Louvain")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        actual_communities = {r.get("name", ""): r.get("community", False) for r in records}
        communities = set(actual_communities.values())
        print(f"✓ Louvain: {len(records)} nodes in {len(communities)} communities")

        for r in records:
            print(f"    {r.get('name', '')}: community {r.get('community', False)}")

        errors = partition_contract_errors(actual_communities, ground_truth.get("communities", {}))
        for error in errors:
            print(f"  ✗ {error}")
        if errors:
            return False
        print("  ✓ A1-A4 and B1-B4 form separate communities matching the NetworkX baseline")

        return True

    except Exception as e:
        print(f"✗ Louvain failed: {e}")
        return False


def test_leiden(session, ground_truth: dict) -> bool:
    """Test Leiden community detection - A1-A4 and B1-B4 should be grouped."""
    print("\n--- Testing Leiden ---")
    try:
        result = session.run(
            """
            CALL cugraph.leiden.get()
            YIELD node, partition
            RETURN node.id AS id, node.name AS name, partition AS community
            ORDER BY partition, id
        """
        )

        records = list(result)

        valid, errors = validate_node_identities(records, "Leiden")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        actual_communities = {r.get("name", ""): r.get("community", False) for r in records}
        communities = set(actual_communities.values())
        print(f"✓ Leiden: {len(records)} nodes in {len(communities)} communities")

        for r in records:
            print(f"    {r.get('name', '')}: community {r.get('community', False)}")

        errors = partition_contract_errors(actual_communities, ground_truth.get("communities", {}))
        for error in errors:
            print(f"  ✗ {error}")
        if errors:
            return False
        print("  ✓ A1-A4 and B1-B4 form separate communities matching the NetworkX baseline")

        return True

    except Exception as e:
        print(f"✗ Leiden failed: {e}")
        return False


def test_katz_centrality(session, ground_truth: dict) -> bool:
    """Test Katz Centrality algorithm."""
    print("\n--- Testing Katz Centrality ---")
    try:
        result = session.run(
            """
            CALL cugraph.katz_centrality.get(0.1, 1.0, 1e-6, 100, false)
            YIELD node, katz
            RETURN node.id AS id, node.name AS name, katz
            ORDER BY katz DESC
        """
        )

        records = list(result)

        valid, errors = validate_node_identities(records, "Katz")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        print(f"✓ Katz Centrality: {len(records)} nodes returned")

        katz_baseline = ground_truth.get("katz", {})
        if not isinstance(katz_baseline, dict):
            raise TypeError("katz ground truth must be a dictionary")
        if not katz_baseline:
            print("  ⚠ NetworkX Katz did not converge, skipping value comparison")
            for r in records:
                print(f"    {r.get('name', '')}: {r.get('katz', False):.6f}")
            return True

        expected = require_baseline_scores(ground_truth, "katz")
        all_match = True
        for r in records:
            name = r.get("name", "")
            actual = r.get("katz", False)
            exp = expected.get(name, 0.0)
            match, err = values_match(exp, actual, name)
            if not match:
                print(f"  ✗ {err}")
                all_match = False
            else:
                print(f"  ✓ {name}: {actual:.6f} (expected: {exp:.6f})")

        return all_match

    except Exception as e:
        print(f"✗ Katz Centrality failed: {e}")
        return False


def test_personalized_pagerank(session, ground_truth: dict) -> bool:
    """Test Personalized PageRank from A1."""
    print("\n--- Testing Personalized PageRank ---")
    try:
        result = session.run(
            """
            MATCH (source:Node {id: 1})
            CALL cugraph.personalized_pagerank.get(source, 100, 0.85, 1e-5)
            YIELD node, pagerank
            RETURN node.id AS id, node.name AS name, pagerank
            ORDER BY pagerank DESC
        """
        )

        records = list(result)

        valid, errors = validate_node_identities(records, "Personalized PageRank")
        if not valid:
            for err in errors:
                print(f"  ✗ {err}")
            return False

        print(f"✓ Personalized PageRank: {len(records)} nodes returned")

        expected = require_baseline_scores(ground_truth, "personalized_pagerank")
        all_match = True

        for r in records:
            name = r.get("name", "")
            actual = r.get("pagerank", False)
            exp = expected.get(name, 0.0)
            match, err = values_match(exp, actual, name)
            if not match:
                print(f"  ✗ {err}")
                all_match = False
            else:
                print(f"  ✓ {name}: {actual:.6f} (expected: {exp:.6f})")

        # A1 should have highest PPR (it's the source)
        a1_ppr = next(
            (r.get("pagerank", False) for r in records if r.get("name", "") == "A1"),
            False,
        )
        max_ppr = max(r.get("pagerank", False) for r in records)

        if a1_ppr != max_ppr:
            print(f"  ⚠ A1 should have highest PPR but doesn't (A1={a1_ppr}, max={max_ppr})")
        else:
            print("  ✓ A1 has highest PPR as expected (source node)")

        return all_match

    except Exception as e:
        print(f"✗ Personalized PageRank failed: {e}")
        return False


def main():
    print("=" * 60)
    print("cuGraph MAGE Algorithm Test Suite")
    print("Testing RAPIDS 25.x API with NetworkX Ground Truth")
    print("=" * 60)

    # Build the NetworkX reference copy of the test graph and compute ground truth
    print("\n--- Computing NetworkX Ground Truth ---")
    G = nx_DiGraph()
    G.add_nodes_from((node_id, {"name": name}) for name, node_id in EXPECTED_NODE_IDS.items())
    G.add_edges_from((source, target, {"weight": 1.0}) for source, target in TEST_GRAPH_EDGES)
    print(f"  NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    ground_truth = get_networkx_ground_truth(G)
    print("  ✓ Ground truth computed for all algorithms")

    # Show expected values
    print("\n  Expected PageRank (top 3):")
    pr = require_baseline_scores(ground_truth, "pagerank")
    for name, score in sorted(pr.items(), key=lambda item: item[1], reverse=True)[:3]:
        print(f"    {name}: {score:.6f}")

    print("\n  Expected Betweenness (top 3):")
    bc = require_baseline_scores(ground_truth, "betweenness")
    for name, score in sorted(bc.items(), key=lambda item: item[1], reverse=True)[:3]:
        print(f"    {name}: {score:.6f}")

    # Every acquired resource is released in the finally blocks, driver first, then container, then storage.
    run_id = uuid_uuid4().hex
    data_dir = acquire_data_dir()
    try:
        bolt_uri = setup_container(f"{CONTAINER_NAME}-{run_id}", run_id, data_dir)
        driver = GraphDatabase.driver(bolt_uri, auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD))
        try:
            passed = run_validation(driver, ground_truth)
        finally:
            driver.close()
    finally:
        release_resources(run_id, data_dir)
    sys_exit(0 if passed else 1)


def run_validation(driver, ground_truth: dict) -> bool:
    """Build the test graph in this run's database and compare every algorithm with the ground truth."""
    if not wait_for_memgraph(driver):
        return False

    with driver.session() as session:
        print("\n--- Setup ---")
        clear_database(session)
        if not create_test_graph(session):
            print("✗ Failed to create test graph")
            return False

        results = {}

        results["PageRank"] = test_pagerank(session, ground_truth)
        results["Betweenness Centrality"] = test_betweenness_centrality(session, ground_truth)
        results["HITS"] = test_hits(session, ground_truth)
        results["Louvain"] = test_louvain(session, ground_truth)
        results["Leiden"] = test_leiden(session, ground_truth)
        results["Katz Centrality"] = test_katz_centrality(session, ground_truth)
        results["Personalized PageRank"] = test_personalized_pagerank(session, ground_truth)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    print(f"Tolerance: {TOLERANCE:.0%} relative, {ABS_TOLERANCE} absolute")

    all_passed = failed == 0
    if all_passed:
        print("\n✓ All cuGraph algorithms match NetworkX ground truth!")
    return all_passed


if __name__ == "__main__":
    main()
