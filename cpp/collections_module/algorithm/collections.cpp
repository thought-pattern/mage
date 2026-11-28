#include "collections.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

bool ContainsNull(const mgp::Value &value) {
  if (value.IsNull()) {
    return true;
  }
  if (value.IsList()) {
    for (const auto &element : value.ValueList()) {
      if (ContainsNull(element)) {
        return true;
      }
    }
  } else if (value.IsMap()) {
    for (const auto &item : value.ValueMap()) {
      if (ContainsNull(item.value)) {
        return true;
      }
    }
  }
  return false;
}

int64_t DoubleToCheckedInt64(double d) {
  constexpr double kMinInt64AsDouble = -9223372036854775808.0;
  constexpr double kExclusiveMaxInt64AsDouble = 9223372036854775808.0;
  if (!std::isfinite(d) || d < kMinInt64AsDouble || d >= kExclusiveMaxInt64AsDouble) {
    throw mgp::ValueException("Floating-point value is outside the signed 64-bit integer range.");
  }
  return static_cast<int64_t>(d);
}

std::string TypeTag(mgp::Type type) {
  std::ostringstream oss;
  oss << type;
  return oss.str();
}

std::string StructuralKey(const mgp::Value &value) {
  std::string out = TypeTag(value.Type());
  out.push_back(':');
  if (value.IsNull()) {
    return out;
  }
  if (value.IsList()) {
    out.push_back('[');
    for (const auto item : value.ValueList()) {
      const std::string sub = StructuralKey(item);
      out += std::to_string(sub.size());
      out.push_back('#');
      out += sub;
    }
    out.push_back(']');
    return out;
  }
  if (value.IsMap()) {
    std::map<std::string, std::string> sorted_items;
    for (const auto item : value.ValueMap()) {
      sorted_items.emplace(std::string(item.key), StructuralKey(item.value));
    }
    out.push_back('{');
    for (const auto &[k, v] : sorted_items) {
      out += std::to_string(k.size());
      out.push_back(':');
      out += k;
      out.push_back('=');
      out += std::to_string(v.size());
      out.push_back('#');
      out += v;
    }
    out.push_back('}');
    return out;
  }
  const std::string rendered = value.ToString();
  out += std::to_string(rendered.size());
  out.push_back(':');
  out += rendered;
  return out;
}

mgp::Map BuildFrequencyMap(const mgp::List &input_list) {
  std::unordered_map<mgp::Value, int64_t> frequency_map;
  for (auto &&element : input_list) {
    frequency_map[element]++;
  }

  struct BucketEntry {
    std::string type_tag;
    std::string structural_key;
    int64_t count{0};
  };

  std::map<std::string, std::vector<BucketEntry>> grouped;
  for (const auto &[element, count] : frequency_map) {
    const std::string base_key = element.IsNull() ? std::string("NO_VALUE") : element.ToString();
    grouped[base_key].push_back(BucketEntry{TypeTag(element.Type()), StructuralKey(element), count});
  }

  mgp::Map result_map;
  std::unordered_set<std::string> used_keys;
  for (const auto &[base_key, entries] : grouped) {
    if (entries.size() == 1) {
      used_keys.insert(base_key);
      result_map.Insert(base_key, mgp::Value(entries[0].count));
    }
  }
  for (auto &[base_key, entries] : grouped) {
    if (entries.size() <= 1) {
      continue;
    }
    std::sort(entries.begin(), entries.end(), [](const BucketEntry &a, const BucketEntry &b) {
      if (a.type_tag != b.type_tag) {
        return a.type_tag < b.type_tag;
      }
      return a.structural_key < b.structural_key;
    });
    for (const auto &entry : entries) {
      std::string key = entry.type_tag + ":" + base_key;
      size_t ordinal = 2;
      while (used_keys.contains(key)) {
        key = entry.type_tag + "#" + std::to_string(ordinal) + ":" + base_key;
        ++ordinal;
      }
      used_keys.insert(key);
      result_map.Insert(key, mgp::Value(entry.count));
    }
  }
  return result_map;
}

}  // namespace

void Collections::SetResult(mgp::Result &result, const mgp::Value &value) {
  switch (value.Type()) {
    case mgp::Type::Null:
      return result.SetValue();
    case mgp::Type::Bool:
      return result.SetValue(value.ValueBool());
    case mgp::Type::Int:
      return result.SetValue(value.ValueInt());
    case mgp::Type::Double:
      return result.SetValue(value.ValueDouble());
    case mgp::Type::String:
      return result.SetValue(value.ValueString());
    case mgp::Type::List:
      return result.SetValue(value.ValueList());
    case mgp::Type::Map:
      return result.SetValue(value.ValueMap());
    case mgp::Type::Node:
      return result.SetValue(value.ValueNode());
    case mgp::Type::Relationship:
      return result.SetValue(value.ValueRelationship());
    case mgp::Type::Path:
      return result.SetValue(value.ValuePath());
    case mgp::Type::Date:
      return result.SetValue(value.ValueDate());
    case mgp::Type::LocalTime:
      return result.SetValue(value.ValueLocalTime());
    case mgp::Type::LocalDateTime:
      return result.SetValue(value.ValueLocalDateTime());
    case mgp::Type::Duration:
      return result.SetValue(value.ValueDuration());
    case mgp::Type::ZonedDateTime:
      return result.SetValue(value.ValueZonedDateTime());
    case mgp::Type::Point2d:
      return result.SetValue(value.ValuePoint2d());
    case mgp::Type::Point3d:
      return result.SetValue(value.ValuePoint3d());
    case mgp::Type::Enum:
      return result.SetValue(value.ValueEnum());

    default:
        std::ostringstream oss;
        oss << value.Type();
        throw mgp::ValueException("No Result.SetValue for: " + oss.str());
  }
}

void Collections::SumLongs(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    if (arguments[0].IsNull()) {
      result.SetValue();
      return;
    }
    int64_t sum{0};
    const auto list{arguments[0].ValueList()};

    for (const auto list_item : list) {
      if (!list_item.IsNumeric()) {
        std::ostringstream oss;
        oss << list_item.Type();
        throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
      }
      const int64_t addend = list_item.IsInt() ? list_item.ValueInt() : DoubleToCheckedInt64(list_item.ValueDouble());
      if ((addend > 0 && sum > std::numeric_limits<int64_t>::max() - addend) ||
          (addend < 0 && sum < std::numeric_limits<int64_t>::min() - addend)) {
        throw mgp::ValueException("Signed 64-bit integer overflow in sum_longs.");
      }
      sum += addend;
    }
    result.SetValue(sum);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Avg(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    if (arguments[0].IsNull()) {
      result.SetValue();
      return;
    }
    const auto list{arguments[0].ValueList()};
    if (list.Empty()) {
      result.SetValue();
      return;
    }
    double sum{0.0};
    bool all_finite{true};
    const size_t count = list.Size();

    for (const auto list_item : list) {
      if (!list_item.IsNumeric()) {
        std::ostringstream oss;
        oss << list_item.Type();
        throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
      }
      const double val = list_item.ValueNumeric();
      if (!std::isfinite(val)) {
        all_finite = false;
      }
      sum += val;
    }

    double average{0.0};
    if (!std::isfinite(sum) && all_finite) {
      const int scale_exp = static_cast<int>(std::bit_width(count));
      double scaled_sum{0.0};
      for (const auto list_item : list) {
        scaled_sum += std::ldexp(list_item.ValueNumeric(), -scale_exp);
      }
      const double scaled_avg = scaled_sum / static_cast<double>(count);
      average = std::ldexp(scaled_avg, scale_exp);
      if (!std::isfinite(average)) {
        average = std::copysign(std::numeric_limits<double>::max(), scaled_avg);
      }
    } else {
      average = sum / static_cast<double>(count);
    }

    result.SetValue(average);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::ContainsAll(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    if (arguments[0].IsNull() || arguments[1].IsNull()) {
      result.SetValue(false);
      return;
    }
    const auto list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set(list1.begin(), list1.end());

    const auto list2{arguments[1].ValueList()};

    std::unordered_set<mgp::Value> values(list2.begin(), list2.end());

    result.SetValue(
        std::all_of(values.begin(), values.end(), [&set](const auto &x) { return !ContainsNull(x) && set.contains(x); }));

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Intersection(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    const auto list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set1(list1.begin(),list1.end());

    const auto list2{arguments[1].ValueList()};
    std::unordered_set<mgp::Value> set2(list2.begin(),list2.end());


    if (set1.size() > set2.size()) {
      std::swap(set1, set2);
    }

    mgp::List intersection{};
    for (const auto &element : set1) {
      if (set2.contains(element)) {
        intersection.AppendExtend(std::move(element));
      }
    }

    result.SetValue(intersection);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::RemoveAll(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    if (arguments[0].IsNull()) {
      result.SetValue();
      return;
    }
    if (arguments[1].IsNull()) {
      result.SetValue(arguments[0].ValueList());
      return;
    }
    const auto input_list = arguments[0].ValueList();
    const auto to_remove_list = arguments[1].ValueList();

    const std::unordered_set<mgp::Value> to_remove(to_remove_list.begin(), to_remove_list.end());

    mgp::List final_list = mgp::List();
    for (const auto &element : input_list) {
      if (!to_remove.contains(element)) {
        final_list.AppendExtend(element);
      }
    }

    result.SetValue(final_list);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Sum(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    double sum{0};
    const auto list = arguments[0].ValueList();

    for (const auto value : list) {
      if (!value.IsNumeric()) {
        throw std::invalid_argument("One of the list elements is not a number.");
      }
      sum += value.ValueNumeric();
    }

    result.SetValue(sum);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Union(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    const auto list1 = arguments[0].ValueList();
    const auto list2 = arguments[1].ValueList();

    std::unordered_map<int64_t, std::vector<mgp::Value>> unionMap;

    for (const auto value : list1) {
      if (auto search = unionMap.find(std::hash<mgp::Value>{}(value)); search != unionMap.end()) {
        if (std::find(search->second.begin(), search->second.end(), value) != search->second.end()) {
          continue;
        }
        search->second.push_back(value);
      }
      unionMap.insert({std::hash<mgp::Value>{}(value), std::vector<mgp::Value>{value}});
    }
    for (const auto value : list2) {
      if (auto search = unionMap.find(std::hash<mgp::Value>{}(value)); search != unionMap.end()) {
        if (std::find(search->second.begin(), search->second.end(), value) != search->second.end()) {
          continue;
        }
        search->second.push_back(value);
      }
      unionMap.insert({std::hash<mgp::Value>{}(value), std::vector<mgp::Value>{value}});
    }

    mgp::List unionList;

    for (auto pair : unionMap) {
      for (auto value : pair.second) {
        unionList.AppendExtend(std::move(value));
      }
    }

    result.SetValue(unionList);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Sort(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    const auto list = arguments[0].ValueList();
    std::vector<mgp::Value> sorted;

    for (const auto value : list) {
      sorted.push_back(std::move(value));
    }

    std::sort(sorted.begin(), sorted.end());

    result.SetValue(mgp::List(std::move(sorted)));

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::ContainsSorted(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    if (arguments[0].IsNull()) {
      result.SetValue(false);
      return;
    }
    bool contains{false};
    const auto list = arguments[0].ValueList();
    const auto element = arguments[1];
    if (list.Empty() || (!element.IsNull() && ContainsNull(element))) {
      result.SetValue(false);
      return;
    }

    size_t left{0};
    size_t right{list.Size() - 1};

    while (left <= right) {
      const size_t check = left + (right - left) / 2;
      if (!element.IsNull() && list[check] == element) {
        contains = true;
        break;
      } else if (list[check].IsNull() && element.IsNull()) {
        break;
      } else if (element < list[check]) {
        if (check == 0) {
          break;
        }
        right = check - 1;
      } else {
        if (check == std::numeric_limits<size_t>::max()) {
          break;
        }
        left = check + 1;
      }
    }

    result.SetValue(contains);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Max(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    if (arguments[0].IsNull()) {
      result.SetValue();
      return;
    }
    const auto list = arguments[0].ValueList();

    if (list.Empty()) {
      throw mgp::ValueException("Empty input list.");
    }

    mgp::Value max{};
    for (const auto value : list) {
      if (value.IsNull()) {
        continue;
      }
      if (max.IsNull() || max < value) {  // `<` throws if the values can't be compared
        max = value;
      }
    }

    SetResult(result, max);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};

  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto inputList = arguments[0].ValueList();
    const auto delimiter = arguments[1];

    if (inputList.Empty()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Collections::kResultSplit).c_str(), inputList);
      return;
    }

    mgp::List part = mgp::List();
    for (const auto value : inputList) {
      if (value != delimiter) {
        part.AppendExtend(value);
        continue;
      }
      if (part.Empty()) {
        continue;
      }
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Collections::kResultSplit).c_str(), part);
      part = mgp::List();
    }
    if (part.Empty()) {
      return;
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Collections::kResultSplit).c_str(), std::move(part));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Pairs(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    mgp::List pairsList = mgp::List();

    const auto inputList = arguments[0].ValueList();

    if (inputList.Size() == 0) {
      result.SetValue(pairsList);
      return;
    }
    for (size_t i = 0; i < inputList.Size() - 1; i++) {
      mgp::List helper = mgp::List();
      helper.AppendExtend(inputList[i]);
      helper.AppendExtend(inputList[i + 1]);
      pairsList.AppendExtend(mgp::Value(std::move(helper)));
    }
    mgp::List helper = mgp::List();
    helper.AppendExtend(inputList[inputList.Size() - 1]);
    helper.AppendExtend(mgp::Value());
    pairsList.AppendExtend(mgp::Value(std::move(helper)));

    result.SetValue(pairsList);
  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Contains(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    if (arguments[0].IsNull()) {
      result.SetValue(false);
      return;
    }
    const mgp::List &list = arguments[0].ValueList();
    const mgp::Value &value = arguments[1];

    bool contains_value{false};

    if (ContainsNull(value) || list.Empty()) {
      result.SetValue(contains_value);
      return;
    }
    for (size_t i = 0; i < list.Size(); i++) {
      if (list[i] == value) {
        contains_value = true;
        break;
      }
    }
    result.SetValue(contains_value);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::UnionAll(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    mgp::List list1 = arguments[0].ValueList();
    mgp::List list2 = arguments[1].ValueList();

    for (size_t i = 0; i < list2.Size(); i++) {
      list1.AppendExtend(list2[i]);
    }
    result.SetValue(list1);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Min(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    if (arguments[0].IsNull()) {
      result.SetValue();
      return;
    }
    const mgp::List &list = arguments[0].ValueList();
    if (list.Empty()) {
      throw mgp::ValueException("Empty input list");
    }

    mgp::Value min{};
    mgp::Type type{mgp::Type::Null};
    bool is_numeric{false};
    for (size_t i = 0; i < list.Size(); i++) {
      if (list[i].IsNull()) {
        continue;
      }
      if (min.IsNull()) {
        type = list[i].Type();
        if (type == mgp::Type::Map || type == mgp::Type::Path || type == mgp::Type::List) {
          std::ostringstream oss;
          oss << type;
          throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
        }
        is_numeric = list[i].IsNumeric();
        min = list[i];
        continue;
      }
      if (list[i].Type() != type && !(is_numeric && list[i].IsNumeric())) {
        throw mgp::ValueException("All elements must be of the same type!");
      }

      if (list[i] < min) {
        min = list[i];
      }
    }

    SetResult(result, min);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::ToSet(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    mgp::List list = arguments[0].ValueList();
    std::unordered_set<mgp::Value> set(list.begin(),list.end());

    mgp::List return_list;
    for (auto elem : set) {
      return_list.AppendExtend(std::move(elem));
    }
    result.SetValue(std::move(return_list));

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Partition(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List input_list = arguments[0].ValueList();
    const int64_t partition_size = arguments[1].ValueInt();

    int64_t current_size = 0;
    mgp::List temp;
    mgp::List result;
    for (mgp::Value list_value : input_list) {
      if (current_size == 0) {
        temp = mgp::List();
      }
      temp.AppendExtend(std::move(list_value));
      current_size++;

      if (current_size == partition_size) {
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kReturnValuePartition).c_str(), std::move(temp));
        current_size = 0;
      }
    }

    if (current_size != partition_size && current_size != 0) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kReturnValuePartition).c_str(), std::move(temp));
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

namespace {

// Helper function to recursively flatten a list
void FlattenHelper(const mgp::Value &value, mgp::List &result) {
    if (value.IsNull()) {
        // Skip null values
        return;
    }

    if (value.IsList()) {
        auto list = value.ValueList();
        for (const auto &item : list) {
            FlattenHelper(item, result);
        }
    } else {
        result.AppendExtend(value);
    }
}

}  // namespace

void Collections::Flatten(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
    mgp::MemoryDispatcherGuard guard{memory};
    const auto arguments = mgp::List(args);
    auto result = mgp::Result(res);

    try {
        if (arguments.Size() != 1) {
            throw mgp::ValueException("The procedure expects 1 argument, got " + std::to_string(arguments.Size()));
        }

        const auto &input = arguments[0];
        if (!input.IsList()) {
            throw mgp::ValueException("The argument must be a list");
        }

        // Create result list
        mgp::List flattened;

        // Directly flatten the input (handles null and lists recursively)
        FlattenHelper(input, flattened);

        result.SetValue(flattened);

    } catch (const std::exception &e) {
        result.SetErrorMessage(e.what());
        return;
    }
}

void Collections::FrequenciesAsMap(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto const arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    if (arguments[0].IsNull()) {
      result.SetValue(mgp::Map());
      return;
    }
    auto const input_list = arguments[0].ValueList();
    mgp::Map result_map = BuildFrequencyMap(input_list);
    result.SetValue(std::move(result_map));

  } catch (std::exception const &e) {
    result.SetErrorMessage(e.what());
  }
}
