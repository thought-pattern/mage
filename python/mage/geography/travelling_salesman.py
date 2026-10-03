"""Utilities for travelling salesman."""

from itertools import combinations as itertools_combinations
from sys import stderr as sys_stderr
from sys import version as sys_version

from numpy import ndarray as np_ndarray
from numpy import shape as np_shape
from numpy import zeros as np_zeros

from mage.geography import calculate_distance_between_points

try:
    from networkx import Graph as nx_Graph
    from networkx import MultiGraph as nx_MultiGraph
    from networkx import dfs_preorder_nodes as nx_dfs_preorder_nodes
    from networkx import eulerian_path as nx_eulerian_path
    from networkx import max_weight_matching as nx_max_weight_matching
    from networkx import minimum_spanning_tree as nx_minimum_spanning_tree
except ImportError as import_error:
    sys_stderr.write(f"NOTE: Please install networkx to be able touse graph_analyzer module. Using Python: {sys_version}")
    raise import_error from import_error


def create_distance_matrix(points: list[dict[str, float]]):
    """
    Creates a quadratic matrix of distances between points.
    :param points: List of dictionaries with lat and lng coordinates
    :return: Distance matrix
    """

    distance_matrix = np_zeros([len(points), len(points)])

    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            d = calculate_distance_between_points(points[i], points[j])
            if d is False:
                return False
            distance_matrix[i][j] = distance_matrix[j][i] = d

    return distance_matrix


def solve_2_approx(dm: np_ndarray):
    """
    Solves the tsp_module problem with 2-approximation.
    :param dm: Distance matrix.
    :return: List of indices - path between them (based on distance matrix indexes)
    """

    point_count = tour_point_count(dm)
    if point_count < 2:
        path = list(range(point_count))
        return path

    mst = get_mst(dm)
    path = [x for x in nx_dfs_preorder_nodes(mst)]
    path.append(path[0])

    return path


def solve_1_5_approx(dm: np_ndarray):
    """
    Solves the tsp_module problem with 1.5-approximation (Christofides algorithm). The bound holds for metric
    distances, which the geographic distance matrix is.
    :param distance_matrix: Distance matrix.
    :return: List of indices - path between them (based on distance matrix indexes)
    """

    point_count = tour_point_count(dm)
    if point_count < 2:
        path = list(range(point_count))
        return path

    mst = get_mst(dm)
    odd_vertices = [vertex for vertex, degree in mst.degree if degree % 2 == 1]
    matches = get_minimum_weight_perfect_matching(odd_vertices, dm)

    all_edges = list(mst.edges)
    all_edges.extend(matches)

    euler_circuit = get_euler_circuit(all_edges)
    path = get_hamiltonian_circuit(euler_circuit)

    return path


def solve_greedy(dm: np_ndarray):
    """
    Solves the tsp_module problem with greedy method of taking the closest node to the last.
    :param distance_matrix: Distance matrix.
    :return: List of indices - path between them (based on distance matrix indexes)
    """

    point_count = tour_point_count(dm)
    if point_count < 2:
        path = list(range(point_count))
        return path

    path = []
    visited_vert = dict()
    path.append(0)
    visited_vert[0] = True

    while len(path) != len(dm):
        last = path[-1]
        min_index, min_val = -1, -1

        for i in range(len(dm)):
            value = dm[last][i]
            if last != i and (min_index == -1 or min_val > value) and i not in visited_vert:
                min_index = i
                min_val = value

        path.append(min_index)
        visited_vert[min_index] = True

    path.append(0)

    return path


def tour_point_count(dm: np_ndarray) -> int:
    """
    Admits a distance matrix at the solver boundary and returns its point count. Every solver returns a closed tour
    (first index repeated at the end) for two or more points; fewer points have no edge to travel, so the empty
    collection has the empty tour and a single point is the one-index tour [0].
    :param dm: Distance matrix.
    :return: Number of points
    """

    matrix_shape = np_shape(dm)
    if len(matrix_shape) != 2 or matrix_shape[0] != matrix_shape[1]:
        raise ValueError(f"TSP distance matrix must be square, received shape {matrix_shape}")

    point_count = matrix_shape[0]
    return point_count


def get_hamiltonian_circuit(euler_circuit):
    """
    Deletes duplicates of the Euler circuit in order to form hamiltonian circuit where no vertex is
    visited twice or more times
    :param euler_circuit: Eulerian path
    :return:
    """

    path = []
    [path.append(x[0]) for x in euler_circuit]
    path = list(dict.fromkeys(path))
    path.append(path[0])

    return path


def get_euler_circuit(tum_edges):
    """
    Uses nx library for finding an Eulerian circuit
    :param tum_edges: Union of mst and matchings edges
    :return: Eulerian path generator
    """

    g = nx_MultiGraph()

    for edge in tum_edges:
        g.add_edge(edge[0], edge[1])

    path = nx_eulerian_path(g, source=tum_edges[0][0])

    return path


def get_minimum_weight_perfect_matching(odd_vertices: list[int], dm: np_ndarray):
    """
    Minimum-cost perfect matching of the spanning tree's odd-degree vertices, which the Christofides bound requires.
    A maximum-cardinality matching of the complete graph on an even vertex count is perfect, so maximising
    (heaviest + 1 - distance) among those matchings minimises their total distance; the offset keeps every weight
    positive. The transform is stated here instead of using networkx.min_weight_matching, whose objective differs
    between the pinned 2.8 release (reciprocal weights) and 3.x.
    :param odd_vertices: Vertices with odd degree in the minimum spanning tree (always an even count)
    :param dm: Distance matrix
    :return: List of matched edges
    """

    vertex_pairs = list(itertools_combinations(odd_vertices, 2))
    heaviest = max((dm[u][v] for u, v in vertex_pairs), default=0.0)

    matching_graph = nx_Graph()
    for u, v in vertex_pairs:
        matching_graph.add_edge(u, v, weight=heaviest + 1.0 - dm[u][v])

    matched_edges = list(nx_max_weight_matching(matching_graph, maxcardinality=True))

    return matched_edges


def get_mst(dm: np_ndarray):
    """
    Creates the minimum spanning tree using nx.
    :param dm: Distance matrix
    :return: Minimum spanning tree
    """

    g = nx_Graph()

    for i in range(len(dm) - 1):
        for j in range(i + 1, len(dm)):
            g.add_edge(i, j, weight=dm[i][j])

    mst = nx_minimum_spanning_tree(g)

    return mst
