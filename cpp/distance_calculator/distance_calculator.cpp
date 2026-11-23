#include <algorithm>
#include <cmath>
#include <mgp.hpp>
#define KM_MUL 0.001

const char *kProcedureSingle = "single";
const char *kProcedureMultiple = "multiple";

const char *kReturnDistance = "distance";
const char *kReturnDistances = "distances";

const char *kArgumentStartPoint = "start_point";
const char *kArgumentEndPoint = "end_point";
const char *kArgumentStart = "start";
const char *kArgumentEnd = "end";

const char *kArgumentDecimals = "decimals";
const char *kArgumentMetrics = "metrics";

const char *kDefaultArgumentMetrics = "m";
const int64_t kDefaultArgumentDecimals = 2;

const double pi_rad = M_PI / 180.0;
const double R = 6371000.0;

double round_to_decimals(double value, int64_t decimals) {
  if (decimals >= 0) {
    const double factor = std::pow(10.0, static_cast<double>(decimals));
    if (!std::isfinite(factor) || factor == 0.0) {
      return value;
    }
    return std::round(value * factor) / factor;
  }
  const double divisor = std::pow(10.0, -static_cast<double>(decimals));
  if (!std::isfinite(divisor) || divisor == 0.0) {
    return 0.0;
  }
  return std::round(value / divisor) * divisor;
}

double haversine_distance(double lat1, double lng1, double lat2, double lng2, bool use_km, int64_t decimals) {
  double phi_1 = lat1 * pi_rad;
  double phi_2 = lat2 * pi_rad;

  double delta_phi = (lat2 - lat1) * pi_rad;
  double delta_lambda = (lng2 - lng1) * pi_rad;

  double sin_delta_phi = std::sin(delta_phi / 2.0);
  double sin_delta_lambda = std::sin(delta_lambda / 2.0);

  double a = sin_delta_phi * sin_delta_phi + std::cos(phi_1) * std::cos(phi_2) * (sin_delta_lambda * sin_delta_lambda);
  a = std::clamp(a, 0.0, 1.0);
  double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
  double distance = R * c;
  if (use_km) {
    distance *= KM_MUL;
  }
  return round_to_decimals(distance, decimals);
}

double distance_calc(const mgp::Node &node1, const mgp::Node &node2, bool use_km, int64_t decimals) {
  double lat1 = node1.GetProperty("lat").ValueNumeric();
  double lng1 = node1.GetProperty("lng").ValueNumeric();

  double lat2 = node2.GetProperty("lat").ValueNumeric();
  double lng2 = node2.GetProperty("lng").ValueNumeric();

  return haversine_distance(lat1, lng1, lat2, lng2, use_km, decimals);
}

void Single(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const mgp::List arguments = mgp::List(args);
    const mgp::Node &node1 = arguments[0].ValueNode();
    const mgp::Node &node2 = arguments[1].ValueNode();
    const std::string_view metrics = arguments[2].ValueString();
    int64_t decimals = arguments[3].ValueInt();

    mgp::Record record = record_factory.NewRecord();

    record.Insert(kReturnDistance, distance_calc(node1, node2, metrics == "km", decimals));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Multiple(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto arguments = mgp::List(args);
    mgp::List distances = mgp::List();

    const mgp::List start_points = arguments[0].ValueList();
    const mgp::List end_points = arguments[1].ValueList();
    const std::string_view metrics = arguments[2].ValueString();
    int64_t decimals = arguments[3].ValueInt();

    if (start_points.Size() != end_points.Size()) {
      throw mgp::ValueException("Both arrays must be of equal length.");
    }
    
    bool use_km = metrics == "km";
    for (decltype(start_points.Size()) i = 0; i < start_points.Size(); i++) {
      const mgp::Node &node1 = start_points[i].ValueNode();
      const mgp::Node &node2 = end_points[i].ValueNode();
      distances.AppendExtend(mgp::Value(distance_calc(node1, node2, use_km, decimals)));
    }

    mgp::Record record = record_factory.NewRecord();
    record.Insert(kReturnDistances, distances);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(
        Single, kProcedureSingle, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentStartPoint, mgp::Type::Node), mgp::Parameter(kArgumentEndPoint, mgp::Type::Node),
         mgp::Parameter(kArgumentMetrics, mgp::Type::String, kDefaultArgumentMetrics),
         mgp::Parameter(kArgumentDecimals, mgp::Type::Int, kDefaultArgumentDecimals)},
        {mgp::Return(kReturnDistance, mgp::Type::Double)}, module, memory);

    const auto multiple_input = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto multiple_return = std::make_pair(mgp::Type::List, mgp::Type::Double);
    AddProcedure(Multiple, kProcedureMultiple, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentStart, multiple_input), mgp::Parameter(kArgumentEnd, multiple_input),
                  mgp::Parameter(kArgumentMetrics, mgp::Type::String, kDefaultArgumentMetrics),
                  mgp::Parameter(kArgumentDecimals, mgp::Type::Int, kDefaultArgumentDecimals)},
                 {mgp::Return(kReturnDistances, multiple_return)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
