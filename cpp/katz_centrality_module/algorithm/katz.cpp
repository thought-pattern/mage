#include "katz.hpp"

#include <limits>
#include <mutex>
#include <queue>

namespace katz_alg {
namespace {

class KatzCentralityData {
 public:
  void Clear() {
    centralities.clear();
    omegas.clear();
    lr.clear();
    ur.clear();
    active_nodes.clear();
    iteration = 0;
  }

  template <typename T>
  void Init(const T &graph) {
    Clear();

    std::unordered_map<std::uint64_t, double> centrality;
    InitVertexMap(centrality, 0.0, graph);
    centralities.emplace_back(std::move(centrality));

    std::unordered_map<std::uint64_t, double> omega;
    InitVertexMap(omega, 1.0, graph);
    omegas.emplace_back(std::move(omega));

    InitVertexMap(lr, 0.0, graph);
    InitVertexMap(ur, 0.0, graph);
  }

  template <typename T>
  void AddIteration(const T &graph) {
    iteration++;

    std::unordered_map<std::uint64_t, double> centrality;
    InitVertexMap(centrality, 0.0, graph);
    centralities.emplace_back(std::move(centrality));

    std::unordered_map<std::uint64_t, double> omega;
    InitVertexMap(omega, 0.0, graph);
    omegas.emplace_back(std::move(omega));
  }

  bool IsInitialized() const { return !centralities.empty() && iteration > 0; }

  std::vector<std::unordered_map<std::uint64_t, double>> centralities;
  std::vector<std::unordered_map<std::uint64_t, double>> omegas;
  std::unordered_map<std::uint64_t, double> lr;
  std::unordered_map<std::uint64_t, double> ur;
  std::set<std::uint64_t> active_nodes;
  std::uint64_t iteration = 0;

 private:
  void InitVertexMap(std::unordered_map<std::uint64_t, double> &map, const double default_value,
                     const mg_graph::GraphView<> &graph) {
    for (const auto &[_v] : graph.Nodes()) {
      const auto v = graph.GetMemgraphNodeId(_v);
      map[v] = default_value;
    }
  }

  void InitVertexMap(std::unordered_map<std::uint64_t, double> &map, const double default_value,
                     const mgp::Graph &graph) {
    for (const auto &node : graph.Nodes()) {
      map[node.Id().AsUint()] = default_value;
    }
  }
};

std::mutex context_mutex;
KatzCentralityData context;
double stored_alpha = 0.2;
double stored_epsilon = 1e-2;

std::vector<std::pair<std::uint64_t, double>> WrapResults(const KatzCentralityData &ctx) {
  return std::vector<std::pair<std::uint64_t, double>>(ctx.centralities[ctx.iteration].begin(),
                                                       ctx.centralities[ctx.iteration].end());
}

std::uint64_t MaxDegree(const mg_graph::GraphView<> &graph) {
  std::uint64_t max_degree = 0;
  for (const auto &node : graph.Nodes()) {
    const auto out_deg = static_cast<std::uint64_t>(graph.Neighbours(node.id).size());
    const auto in_deg = static_cast<std::uint64_t>(graph.InNeighbours(node.id).size());
    max_degree = std::max(max_degree, std::max(out_deg, in_deg));
  }
  return max_degree;
}

std::uint64_t MaxDegree(const mgp::Graph &graph) {
  std::uint64_t max_degree = 0;
  for (const auto &node : graph.Nodes()) {
    std::uint64_t out_degree = 0;
    for (const auto &relationship : node.OutRelationships()) {
      (void)relationship;
      out_degree++;
    }
    std::uint64_t in_degree = 0;
    for (const auto &relationship : node.InRelationships()) {
      (void)relationship;
      in_degree++;
    }
    max_degree = std::max(max_degree, std::max(out_degree, in_degree));
  }
  return max_degree;
}

bool Converged(KatzCentralityData &ctx, const double epsilon) {
  const auto &centrality = ctx.centralities[ctx.iteration];
  const auto &lr = ctx.lr;
  const auto &ur = ctx.ur;
  const std::size_t k_limit = centrality.size();

  std::vector<std::pair<std::uint64_t, double>> active_centrality;
  active_centrality.reserve(ctx.active_nodes.size());
  for (const auto m : ctx.active_nodes) {
    active_centrality.emplace_back(m, centrality.at(m));
  }

  const auto sort_count = std::min<std::size_t>(k_limit, active_centrality.size());
  std::partial_sort(active_centrality.begin(), active_centrality.begin() + static_cast<std::ptrdiff_t>(sort_count),
                    active_centrality.end(),
                    [](std::pair<std::uint64_t, double> a, std::pair<std::uint64_t, double> b) -> bool {
                      return a.second > b.second;
                    });

  if (k_limit > 0 && active_centrality.size() > k_limit) {
    const double cutoff_lr = lr.at(active_centrality[k_limit - 1].first);
    for (std::size_t i = active_centrality.size(); i > k_limit; --i) {
      const auto node_id = active_centrality[i - 1].first;
      if (ur.at(node_id) - epsilon < cutoff_lr) {
        ctx.active_nodes.erase(node_id);
        active_centrality.erase(active_centrality.begin() + static_cast<std::ptrdiff_t>(i - 1));
      }
    }
  }

  if (active_centrality.size() > k_limit) {
    return false;
  }

  const auto size = std::min<std::size_t>(active_centrality.size(), k_limit);
  for (std::size_t i = 1; i < size; i++) {
    const double u = ur.at(active_centrality[i].first);
    const double l = lr.at(active_centrality[i - 1].first);
    if (!std::isfinite(u) || u - epsilon > l) {
      return false;
    }
  }
  return true;
}

double ComputeUpperBound(const double lr_val, const double omega_val, const double alpha,
                         const std::uint64_t iteration, const std::uint64_t deg_max, const std::size_t num_nodes,
                         const double epsilon) {
  if (omega_val <= 0.0 || alpha == 0.0) {
    return lr_val;
  }
  const double alpha_abs = std::abs(alpha);
  const double alpha_deg = alpha_abs * static_cast<double>(deg_max);
  const double step_contrib = std::pow(alpha_abs, static_cast<double>(iteration)) * omega_val;
  if (alpha_deg < 1.0) {
    const double gamma = static_cast<double>(deg_max) / (1.0 - alpha_deg);
    return lr_val + std::pow(alpha_abs, static_cast<double>(iteration + 1)) * omega_val * gamma;
  }
  if (iteration <= num_nodes) {
    return std::numeric_limits<double>::infinity();
  }
  if (alpha_abs < 1.0) {
    const double tail_est = (step_contrib * alpha_abs) / (1.0 - alpha_abs);
    if (tail_est < epsilon) {
      return lr_val + tail_est;
    }
  }
  return std::numeric_limits<double>::infinity();
}

std::vector<std::pair<std::uint64_t, double>> KatzCentralityLoop(KatzCentralityData &ctx,
                                                                 const mg_graph::GraphView<> &graph,
                                                                 const double alpha, const double epsilon,
                                                                 const std::uint64_t deg_max) {
  const std::size_t num_nodes = graph.Nodes().size();
  do {
    ctx.AddIteration(graph);
    const auto iteration = ctx.iteration;
    double max_omega = 0.0;

    for (const auto &[_v] : graph.Nodes()) {
      const auto v = graph.GetMemgraphNodeId(_v);
      for (const auto &[_u, _] : graph.InNeighbours(_v)) {
        const auto u = graph.GetMemgraphNodeId(_u);
        ctx.omegas[iteration][v] += ctx.omegas[iteration - 1][u];
      }
      max_omega = std::max(max_omega, std::abs(ctx.omegas[iteration][v]));
      ctx.centralities[iteration][v] =
          ctx.centralities[iteration - 1][v] + std::pow(alpha, iteration) * ctx.omegas[iteration][v];

      ctx.lr[v] = ctx.centralities[iteration][v];
      ctx.ur[v] =
          ComputeUpperBound(ctx.lr[v], ctx.omegas[iteration][v], alpha, iteration, deg_max, num_nodes, epsilon);
    }

    if (max_omega == 0.0) {
      break;
    }
  } while (!Converged(ctx, epsilon));

  return WrapResults(ctx);
}

std::vector<std::pair<std::uint64_t, double>> KatzCentralityLoop(KatzCentralityData &ctx, const mgp::Graph &graph,
                                                                 const double alpha, const double epsilon,
                                                                 const std::uint64_t deg_max) {
  std::size_t num_nodes = 0;
  for (const auto &node : graph.Nodes()) {
    (void)node;
    ++num_nodes;
  }
  do {
    ctx.AddIteration(graph);
    const auto iteration = ctx.iteration;
    double max_omega = 0.0;

    for (const auto &node : graph.Nodes()) {
      const auto v = node.Id().AsUint();
      for (const auto &in_relationship : node.InRelationships()) {
        const auto u = in_relationship.From().Id().AsUint();
        ctx.omegas[iteration][v] += ctx.omegas[iteration - 1][u];
      }
      max_omega = std::max(max_omega, std::abs(ctx.omegas[iteration][v]));
      ctx.centralities[iteration][v] =
          ctx.centralities[iteration - 1][v] + std::pow(alpha, iteration) * ctx.omegas[iteration][v];

      ctx.lr[v] = ctx.centralities[iteration][v];
      ctx.ur[v] =
          ComputeUpperBound(ctx.lr[v], ctx.omegas[iteration][v], alpha, iteration, deg_max, num_nodes, epsilon);
    }

    if (max_omega == 0.0) {
      break;
    }
  } while (!Converged(ctx, epsilon));

  return WrapResults(ctx);
}

void UpdateLevel(KatzCentralityData &base_ctx, KatzCentralityData &context_new, std::set<std::uint64_t> &affected_nodes,
                 const mgp::Graph &graph, const std::vector<std::pair<std::uint64_t, uint64_t>> &created_relationships,
                 const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_relationships,
                 const std::set<std::uint64_t> &created_relationship_ids, const double alpha) {
  const auto i = context_new.iteration;

  std::queue<std::uint64_t> queue;
  for (const auto v : affected_nodes) {
    queue.push(v);
  }

  for (const auto [id, value] : base_ctx.omegas[i]) {
    context_new.omegas[i][id] = value;
  }

  while (!queue.empty()) {
    const auto v = queue.front();
    queue.pop();

    if (!graph.ContainsNode(mgp::Id::FromUint(v))) continue;

    for (const auto &r : graph.GetNodeById(mgp::Id::FromUint(v)).OutRelationships()) {
      const auto relationship_id = r.Id().AsUint();
      const auto w = r.To().Id().AsUint();

      if (affected_nodes.find(w) == affected_nodes.end()) {
        queue.push(w);
      }
      affected_nodes.emplace(w);

      if (created_relationship_ids.find(relationship_id) != created_relationship_ids.end()) continue;

      context_new.omegas[i][w] += context_new.omegas[i - 1][v] - base_ctx.omegas[i - 1][v];
    }
  }

  for (const auto [v, w] : created_relationships) {
    context_new.omegas[i][w] += context_new.omegas[i - 1][v];
  }

  for (const auto [v, w] : deleted_relationships) {
    context_new.omegas[i][w] -= base_ctx.omegas[i - 1][v];
  }

  for (const auto w : affected_nodes) {
    if (i != 1) {
      base_ctx.centralities[i][w] = base_ctx.centralities[i - 1][w] + std::pow(alpha, i) * context_new.omegas[i][w];
    } else {
      base_ctx.centralities[i][w] += std::pow(alpha, i) * (context_new.omegas[i][w] - base_ctx.omegas[i][w]);
    }
  }
}

bool IsInconsistent(const KatzCentralityData &ctx, const mg_graph::GraphView<> &graph) {
  for (const auto [node_id] : graph.Nodes()) {
    auto external_id = graph.GetMemgraphNodeId(node_id);
    if (ctx.centralities[ctx.iteration].find(external_id) == ctx.centralities[ctx.iteration].end()) {
      return true;
    }
  }

  for (const auto [node_id, _] : ctx.centralities[ctx.iteration]) {
    if (!graph.NodeExists(node_id)) {
      return true;
    }
  }

  return false;
}

}  // namespace

bool NoPreviousData() {
  std::lock_guard<std::mutex> lock(context_mutex);
  return !context.IsInitialized();
}

std::vector<std::pair<std::uint64_t, double>> GetKatz(const mg_graph::GraphView<> &graph) {
  {
    std::lock_guard<std::mutex> lock(context_mutex);
    if (context.IsInitialized()) {
      if (IsInconsistent(context, graph)) {
        throw std::runtime_error(
            "Graph has been modified and is thus inconsistent with cached Katz centrality scores. To update them, "
            "please call set/reset!");
      }
      return WrapResults(context);
    }
  }
  return SetKatz(graph);
}

std::vector<std::pair<std::uint64_t, double>> SetKatz(const mg_graph::GraphView<> &graph, const double alpha,
                                                      const double epsilon) {
  KatzCentralityData local_ctx;
  local_ctx.Init(graph);

  std::vector<std::pair<std::uint64_t, double>> result;
  if (graph.Edges().empty()) {
    result = WrapResults(local_ctx);
  } else {
    const auto deg_max = MaxDegree(graph);
    std::transform(graph.Nodes().begin(), graph.Nodes().end(),
                   std::inserter(local_ctx.active_nodes, local_ctx.active_nodes.end()),
                   [&graph](mg_graph::Node<> node) -> std::uint64_t { return graph.GetMemgraphNodeId(node.id); });
    result = KatzCentralityLoop(local_ctx, graph, alpha, epsilon, deg_max);
  }

  {
    std::lock_guard<std::mutex> lock(context_mutex);
    context = std::move(local_ctx);
    stored_alpha = alpha;
    stored_epsilon = epsilon;
  }
  return result;
}

std::vector<std::pair<std::uint64_t, double>> UpdateKatz(
    const mgp::Graph &graph, const std::vector<std::uint64_t> &created_nodes,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &created_relationships,
    const std::vector<std::uint64_t> &created_relationship_ids, const std::vector<std::uint64_t> &deleted_nodes,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_relationships) {
  std::lock_guard<std::mutex> lock(context_mutex);
  std::uint64_t n_relationships = 0;
  for (const auto &z : graph.Relationships()) {
    (void)z;
    n_relationships++;
  }

  if (n_relationships == 0) {
    context.Init(graph);
    return WrapResults(context);
  }

  const auto deg_max = MaxDegree(graph);
  std::size_t num_nodes = 0;
  for (const auto &node : graph.Nodes()) {
    (void)node;
    ++num_nodes;
  }

  for (const auto v : created_nodes) {
    context.omegas[0][v] = 1.0;
    context.lr[v] = 0.0;
    context.ur[v] = 0.0;
    for (std::uint64_t i = 0; i <= context.iteration; i++) {
      context.centralities[i][v] = 0.0;
    }
  }

  std::set<std::uint64_t> affected_nodes;
  for (const auto [v, w] : created_relationships) {
    affected_nodes.emplace(v);
    affected_nodes.emplace(w);
  }
  for (const auto [v, w] : deleted_relationships) {
    affected_nodes.emplace(v);
    affected_nodes.emplace(w);
  }

  KatzCentralityData context_new;
  context_new.Init(graph);
  std::set<std::uint64_t> created_relationship_ids_set(created_relationship_ids.begin(),
                                                       created_relationship_ids.end());
  for (std::uint64_t i = 0; i < context.iteration; i++) {
    context_new.AddIteration(graph);
    UpdateLevel(context, context_new, affected_nodes, graph, created_relationships, deleted_relationships,
                created_relationship_ids_set, stored_alpha);
  }

  for (std::uint64_t i = 1; i < context_new.iteration + 1; i++) {
    for (const auto [id, value] : context_new.omegas[i]) {
      context.omegas[i][id] = value;
    }
  }

  for (const auto w : affected_nodes) {
    const auto iteration = context.iteration;
    context.lr[w] = context.centralities[iteration][w];
    context.ur[w] = ComputeUpperBound(context.lr[w], context.omegas[iteration][w], stored_alpha, iteration, deg_max,
                                      num_nodes, stored_epsilon);
  }

  for (const auto v : deleted_nodes) {
    for (std::uint64_t i = 0; i <= context.iteration; i++) {
      context.omegas[i].erase(v);
      context.centralities[i].erase(v);
    }
    context.lr.erase(v);
    context.ur.erase(v);
    context.active_nodes.erase(v);
  }

  double min_lr = 0.0;
  if (!context.active_nodes.empty()) {
    min_lr = std::numeric_limits<double>::infinity();
    for (const auto active_node : context.active_nodes) {
      min_lr = std::min(min_lr, context.lr[active_node]);
    }
  }
  for (const auto &node : graph.Nodes()) {
    const auto w = node.Id().AsUint();
    if (context.ur[w] >= (min_lr - stored_epsilon)) {
      context.active_nodes.emplace(w);
    }
  }

  return KatzCentralityLoop(context, graph, stored_alpha, stored_epsilon, deg_max);
}

void Reset() {
  std::lock_guard<std::mutex> lock(context_mutex);
  context.Clear();
}

}  // namespace katz_alg
