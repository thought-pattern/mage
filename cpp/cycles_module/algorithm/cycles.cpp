#include <unordered_set>

#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

#include "cycles.hpp"

namespace cycles_util {

NodeState::NodeState(std::size_t number_of_nodes) {
  visited.resize(number_of_nodes, false);
  parent.resize(number_of_nodes, std::nullopt);
  depth.resize(number_of_nodes, 0);
}

void NodeState::SetVisited(std::uint64_t node_id) { visited[node_id] = true; }

bool NodeState::IsVisited(std::uint64_t node_id) const { return visited[node_id]; }

void NodeState::SetParent(std::uint64_t parent_id, std::uint64_t node_id) { parent[parent_id] = node_id; }

std::optional<std::uint64_t> NodeState::GetParent(std::uint64_t node_id) const { return parent[node_id]; }

void NodeState::SetDepth(std::uint64_t node_id, std::uint64_t node_depth) { depth[node_id] = node_depth; }

std::uint64_t NodeState::GetDepth(std::uint64_t node_id) const { return depth[node_id]; }

void FindNonSpanningTreeEdges(std::uint64_t node_id, const mg_graph::GraphView<> &graph, NodeState *state,
                              std::set<std::pair<uint64_t, uint64_t>> *non_st_edges) {
  struct DfsFrame {
    std::uint64_t current_id;
    std::size_t next_neighbour_idx;
    std::unordered_set<std::uint64_t> unique_neighbour;
  };

  state->SetVisited(node_id);
  std::vector<DfsFrame> stack;
  stack.push_back(DfsFrame{node_id, 0, {}});

  while (!stack.empty()) {
    auto &frame = stack.back();
    const auto current_id = frame.current_id;
    const auto &neighbours = graph.Neighbours(current_id);
    if (frame.next_neighbour_idx >= neighbours.size()) {
      stack.pop_back();
      continue;
    }

    const auto next_id = neighbours[frame.next_neighbour_idx++].node_id;

    const auto parent = state->GetParent(current_id);
    if ((parent && next_id == *parent) || frame.unique_neighbour.find(next_id) != frame.unique_neighbour.end()) {
      continue;
    }

    frame.unique_neighbour.insert(next_id);

    if (state->IsVisited(next_id)) {
      non_st_edges->insert(std::minmax(current_id, next_id));
      continue;
    }

    state->SetParent(next_id, current_id);
    state->SetDepth(next_id, state->GetDepth(current_id) + 1);
    state->SetVisited(next_id);
    stack.push_back(DfsFrame{next_id, 0, {}});
  }
}

void FindFundamentalCycles(const std::set<std::pair<std::uint64_t, std::uint64_t>> &non_st_edges,
                           const NodeState &state, std::vector<std::vector<std::uint64_t>> *fundamental_cycles) {
  for (const auto &[from, to] : non_st_edges) {
    fundamental_cycles->emplace_back(FindFundamentalCycle(from, to, state));
  }
}

std::vector<std::uint64_t> FindFundamentalCycle(std::uint64_t node_a, std::uint64_t node_b, const NodeState &state) {
  std::vector<std::uint64_t> cycle;
  if (state.depth[node_a] < state.depth[node_b]) {
    std::swap(node_a, node_b);
  }

  // climb until a and b reach the same depth:
  //
  //       ()               ()
  //      /  \             /  \.
  //    ()   (b)   -->   (a)  (b)
  //   /  \             /   \.
  // (a)  ()           ()   ()
  cycle.emplace_back(node_a);
  while (state.depth[node_a] > state.depth[node_b]) {
    const auto maybe_parent = state.GetParent(node_a);
    if (!maybe_parent) {
      throw mg_exception::InvalidIDException();
    }
    node_a = *maybe_parent;
    cycle.emplace_back(node_a);
  }

  if (node_a != node_b) {
    throw std::runtime_error("There should be no cross edges in DFS tree of an undirected graph.");
  }

  // Close the cycle
  cycle.emplace_back(cycle[0]);
  return cycle;
}

void CombineCycleSubset(const std::vector<std::size_t> &selected_indices,
                        const std::vector<std::vector<std::pair<std::uint64_t, std::uint64_t>>> &fundamental_edges,
                        std::vector<std::vector<mg_graph::Node<>>> *cycles) {
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> edge_cnt;
  for (const auto cycle_idx : selected_indices) {
    for (const auto &edge : fundamental_edges[cycle_idx]) {
      edge_cnt[edge]++;
    }
  }

  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adj_list;
  std::unordered_set<std::uint64_t> nodes;
  for (const auto &[key, value] : edge_cnt) {
    if (value % 2 == 0) continue;

    const auto [from, to] = key;
    adj_list[from].push_back(to);
    adj_list[to].push_back(from);
    nodes.insert(from);
    nodes.insert(to);
  }
  if (nodes.empty()) {
    return;
  }

  // deg(v) = 2 for all vertices in a cycle.
  for (const auto node : nodes) {
    if (adj_list[node].size() != 2) return;
  }

  std::unordered_set<std::uint64_t> visited;
  std::vector<mg_graph::Node<>> cycle;
  auto curr_node = *nodes.begin();

  while (visited.find(curr_node) == visited.end()) {
    cycle.push_back({curr_node});
    visited.insert(curr_node);
    for (const auto next_node : adj_list[curr_node]) {
      if (visited.find(next_node) == visited.end()) {
        curr_node = next_node;
        break;
      }
    }
  }

  if (visited.size() != nodes.size()) {
    return;
  }

  if (cycle.size() > 2) {
    cycles->emplace_back(std::move(cycle));
  }
}

void CombineCycles(std::uint64_t mask, const std::vector<std::vector<std::uint64_t>> &fundamental_cycles,
                   const mg_graph::GraphView<> &graph, std::vector<std::vector<mg_graph::Node<>>> *cycles) {
  (void)graph;
  std::vector<std::vector<std::pair<std::uint64_t, std::uint64_t>>> fundamental_edges(fundamental_cycles.size());
  std::vector<std::size_t> selected;
  for (std::size_t i = 0; i < fundamental_cycles.size() && i < 64; ++i) {
    if ((mask & (std::uint64_t{1} << i)) == 0) continue;
    selected.push_back(i);
    for (std::size_t j = 1; j < fundamental_cycles[i].size(); ++j) {
      fundamental_edges[i].push_back(std::minmax(fundamental_cycles[i][j], fundamental_cycles[i][j - 1]));
    }
  }
  if (!selected.empty()) {
    CombineCycleSubset(selected, fundamental_edges, cycles);
  }
}

void GetCyclesFromFundamentals(const std::vector<std::vector<uint64_t>> &fundamental_cycles,
                               const mg_graph::GraphView<> &graph, std::vector<std::vector<mg_graph::Node<>>> *cycles) {
  (void)graph;
  const std::size_t num_fundamentals = fundamental_cycles.size();
  if (num_fundamentals == 0) {
    return;
  }

  std::vector<std::vector<std::pair<std::uint64_t, std::uint64_t>>> fundamental_edges(num_fundamentals);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> cycles_by_edge;
  for (std::size_t i = 0; i < num_fundamentals; ++i) {
    auto &edges = fundamental_edges[i];
    edges.reserve(fundamental_cycles[i].size());
    for (std::size_t j = 1; j < fundamental_cycles[i].size(); ++j) {
      const auto edge = std::minmax(fundamental_cycles[i][j], fundamental_cycles[i][j - 1]);
      edges.push_back(edge);
      cycles_by_edge[edge].push_back(i);
    }
  }

  std::vector<std::vector<std::size_t>> fc_adj(num_fundamentals);
  for (const auto &[edge, owners] : cycles_by_edge) {
    (void)edge;
    for (std::size_t a = 0; a < owners.size(); ++a) {
      for (std::size_t b = a + 1; b < owners.size(); ++b) {
        fc_adj[owners[a]].push_back(owners[b]);
        fc_adj[owners[b]].push_back(owners[a]);
      }
    }
  }
  for (auto &neighbours : fc_adj) {
    std::sort(neighbours.begin(), neighbours.end());
    neighbours.erase(std::unique(neighbours.begin(), neighbours.end()), neighbours.end());
  }

  std::vector<bool> comp_visited(num_fundamentals, false);
  for (std::size_t start = 0; start < num_fundamentals; ++start) {
    if (comp_visited[start]) {
      continue;
    }
    std::size_t comp_size = 0;
    std::vector<std::size_t> bfs_stack{start};
    comp_visited[start] = true;
    while (!bfs_stack.empty()) {
      const auto cur = bfs_stack.back();
      bfs_stack.pop_back();
      ++comp_size;
      for (const auto nb : fc_adj[cur]) {
        if (!comp_visited[nb]) {
          comp_visited[nb] = true;
          bfs_stack.push_back(nb);
        }
      }
    }
    if (comp_size >= 64) {
      throw std::overflow_error(
          "Graph component has too many coupled fundamental cycles (>= 64) to enumerate subset combinations.");
    }
  }

  struct SubsetFrame {
    std::size_t next_cand_pos;
    std::vector<std::size_t> candidates;
    std::vector<std::size_t> newly_blocked;
  };

  std::vector<bool> blocked(num_fundamentals, false);
  std::vector<std::size_t> selected;
  for (std::size_t root = 0; root < num_fundamentals; ++root) {
    std::fill(blocked.begin(), blocked.end(), false);
    for (std::size_t prev = 0; prev <= root; ++prev) {
      blocked[prev] = true;
    }

    selected.clear();
    selected.push_back(root);
    CombineCycleSubset(selected, fundamental_edges, cycles);

    std::vector<std::size_t> root_candidates;
    for (const auto nb : fc_adj[root]) {
      if (!blocked[nb]) {
        blocked[nb] = true;
        root_candidates.push_back(nb);
      }
    }
    if (root_candidates.empty()) {
      continue;
    }

    std::vector<SubsetFrame> stack;
    stack.push_back(SubsetFrame{0, std::move(root_candidates), {}});
    while (!stack.empty()) {
      auto &top = stack.back();
      if (top.next_cand_pos >= top.candidates.size()) {
        for (const auto nb : top.newly_blocked) {
          blocked[nb] = false;
        }
        if (selected.size() > 1) {
          selected.pop_back();
        }
        stack.pop_back();
        continue;
      }

      const std::size_t pos = top.next_cand_pos++;
      const std::size_t chosen = top.candidates[pos];
      selected.push_back(chosen);
      CombineCycleSubset(selected, fundamental_edges, cycles);

      std::vector<std::size_t> child_candidates(top.candidates.begin() + static_cast<std::ptrdiff_t>(pos + 1),
                                                top.candidates.end());
      std::vector<std::size_t> newly_blocked;
      for (const auto nb : fc_adj[chosen]) {
        if (!blocked[nb]) {
          blocked[nb] = true;
          newly_blocked.push_back(nb);
          child_candidates.push_back(nb);
        }
      }
      stack.push_back(SubsetFrame{0, std::move(child_candidates), std::move(newly_blocked)});
    }
  }
}

}  // namespace cycles_util

namespace cycles_alg {

std::vector<std::vector<mg_graph::Node<>>> GetCycles(const mg_graph::GraphView<> &graph) {
  auto number_of_nodes = graph.Nodes().size();

  cycles_util::NodeState state(number_of_nodes);

  std::vector<std::vector<mg_graph::Node<>>> cycles;
  // TODO: Solve for each connected component
  for (const auto &node : graph.Nodes()) {
    if (state.IsVisited(node.id)) {
      continue;
    }

    // First we find edges that do not lie on a DFS tree (basically,
    // backedges and crossedges). From those edges we expand into the
    // spanning tree to obtain all fundamental cycles.
    state.SetParent(node.id, node.id);
    std::set<std::pair<std::uint64_t, std::uint64_t>> non_st_edges;
    cycles_util::FindNonSpanningTreeEdges(node.id, graph, &state, &non_st_edges);

    // After finding non spanning-tree edges, we obtain fundamental cycles
    std::vector<std::vector<std::uint64_t>> fundamental_cycles;
    cycles_util::FindFundamentalCycles(non_st_edges, state, &fundamental_cycles);

    // Getting elementary cycles from the subset of fundamental ones
    cycles_util::GetCyclesFromFundamentals(fundamental_cycles, graph, &cycles);
  }

  std::set<std::pair<std::uint64_t, std::uint64_t>> multi_edges;
  for (const auto &edge : graph.Edges()) {
    const auto sorted_edge = std::minmax(edge.from, edge.to);
    if (multi_edges.find(sorted_edge) != multi_edges.end()) {
      continue;
    }
    multi_edges.insert(sorted_edge);

    const auto [from, to] = sorted_edge;
    const auto &edges = graph.GetEdgesBetweenNodes(from, to);
    // Add neighbor cycles and self-loops
    if (edges.size() > 1 || from == to) {
      cycles.push_back({{from}, {to}});
    }
  }

  return cycles;
}

}  // namespace cycles_alg
