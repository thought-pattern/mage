#include "text.hpp"

#include <fmt/args.h>
#include <fmt/format.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <regex>
#include <unordered_map>
#include <vector>
#include <utf8.h>

static std::unordered_map<std::string, std::shared_ptr<const std::regex>> global_regex_cache;
static std::mutex global_regex_cache_mutex;

void Text::Join(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto list{arguments[0].ValueList()};
    const auto delimiter{arguments[1].ValueString()};

    std::string result;
    if (list.Empty()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultJoin).c_str(), result);
      return;
    }

    auto iterator = list.begin();
    result += (*iterator).ValueString();

    for (++iterator; iterator != list.end(); ++iterator) {
      result += delimiter;
      result += (*iterator).ValueString();
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultJoin).c_str(), result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
}

void Text::Format(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    std::string text{arguments[0].ValueString()};
    const auto params{arguments[1].ValueList()};

    fmt::dynamic_format_arg_store<fmt::format_context> storage;
    std::for_each(params.begin(), params.end(),
                  [&storage](const mgp::Value &value) { storage.push_back(value.ToString()); });

    std::string result = fmt::vformat(text, storage);

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFormat).c_str(), result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
}

void Text::RegexGroups(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string input{arguments[0].ValueString()};
    const std::string regex_str{arguments[1].ValueString()};

    std::regex regex{regex_str};

    size_t mark_count = regex.mark_count();
    std::vector<int> submatches;
    for (size_t i = 0; i <= mark_count; ++i) {
      submatches.emplace_back(static_cast<int>(i));
    }

    mgp::List all_results;
    if (regex_str.empty() && input.empty()) {
      all_results.AppendExtend(mgp::Value(mgp::List()));
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultRegexGroups).c_str(), all_results);
      return;
    }

    mgp::List local_results;
    size_t cnt = 1;
    auto words_begin = std::sregex_token_iterator(input.begin(), input.end(), regex, submatches);
    auto words_end = std::sregex_token_iterator();

    for (auto it = words_begin; it != words_end; ++it, ++cnt) {
      local_results.AppendExtend(mgp::Value(it->str()));
      if (cnt % (mark_count + 1) == 0) {
        all_results.AppendExtend(mgp::Value(local_results));
        local_results = mgp::List();
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRegexGroups).c_str(), all_results);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
}

void Text::Replace(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  try {
    auto text = std::string(arguments[0].ValueString());
    const auto regex = std::string(arguments[1].ValueString());
    const auto replacement = std::string(arguments[2].ValueString());

    if (regex.size() == 0) {
      result_obj.SetValue(text);
      return;
    }

    size_t pos = 0;
    while ((pos = text.find(regex, pos)) != std::string::npos) {
      text.replace(pos, regex.length(), replacement);
      pos += replacement.length();
    }

    result_obj.SetValue(std::move(text));
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
    return;
  }
}

void Text::RegReplace(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  try {
    const auto text = std::string(arguments[0].ValueString());
    const auto regex = std::string(arguments[1].ValueString());
    const auto replacement = std::string(arguments[2].ValueString());

    if (regex.size() == 0) {
      result_obj.SetValue(text);
      return;
    }

    // Look up or insert regex into global cache with shared ownership so cache eviction never destroys active patterns
    std::shared_ptr<const std::regex> pattern_ptr;
    {
      std::lock_guard<std::mutex> lock(global_regex_cache_mutex);

      if (global_regex_cache.size() > kMaxRegexCacheSize) {
        global_regex_cache.clear();
      }

      auto it = global_regex_cache.find(regex);
      if (it == global_regex_cache.end()) {
        auto compiled = std::make_shared<const std::regex>(regex);
        it = global_regex_cache.emplace(regex, std::move(compiled)).first;
      }
      pattern_ptr = it->second;
    }

    std::string result_str = std::regex_replace(text, *pattern_ptr, replacement);

    result_obj.SetValue(std::move(result_str));
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
    return;
  }
}


void Text::Distance(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  try {
    // Normalize UTF-8 input and convert to sequences of Unicode code points
    std::string text1_raw = std::string(arguments[0].ValueString());
    std::string text2_raw = std::string(arguments[1].ValueString());

    std::vector<char32_t> text1;
    std::vector<char32_t> text2;

    utf8::utf8to32(text1_raw.begin(), text1_raw.end(), std::back_inserter(text1));
    utf8::utf8to32(text2_raw.begin(), text2_raw.end(), std::back_inserter(text2));

    if (text1.size() < text2.size()) {
      std::swap(text1, text2);
    }
    const size_t m = text1.size();
    const size_t n = text2.size();
    std::vector<size_t> prev(n + 1);
    std::vector<size_t> curr(n + 1);

    for (size_t j = 0; j <= n; j++) {
      prev[j] = j;
    }

    for (size_t i = 1; i <= m; i++) {
      curr[0] = i;
      for (size_t j = 1; j <= n; j++) {
        if (text1[i - 1] == text2[j - 1]) {
          curr[j] = prev[j - 1];
        } else {
          curr[j] = 1 + std::min({prev[j], curr[j - 1], prev[j - 1]});
        }
      }
      prev.swap(curr);
    }

    result_obj.SetValue(static_cast<int64_t>(prev[n]));
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
  }
}

void Text::IndexOf(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  try {
    // Handle nulls
    if (arguments[0].IsNull() || arguments[1].IsNull()) {
      result_obj.SetValue();
      return;
    }
    const std::string_view text_raw = arguments[0].ValueString();
    const std::string_view lookup_raw = arguments[1].ValueString();

    std::vector<char32_t> text_cp;
    std::vector<char32_t> lookup_cp;
    utf8::utf8to32(text_raw.begin(), text_raw.end(), std::back_inserter(text_cp));
    utf8::utf8to32(lookup_raw.begin(), lookup_raw.end(), std::back_inserter(lookup_cp));

    const int64_t text_len = static_cast<int64_t>(text_cp.size());
    int64_t from = 0;
    int64_t to = -1;
    if (arguments.Size() > 2 && !arguments[2].IsNull()) {
      from = arguments[2].ValueInt();
    }
    if (arguments.Size() > 3 && !arguments[3].IsNull()) {
      to = arguments[3].ValueInt();
    }
    if (to == -1 || to > text_len) {
      to = text_len;
    }
    if (from < 0) {
      from = 0;
    }
    if (from > to) {
      from = to;
    }

    int64_t result_index = -1;
    const int64_t lookup_len = static_cast<int64_t>(lookup_cp.size());
    if (from < to && lookup_len > 0 && lookup_len <= (to - from)) {
      const auto window_begin = text_cp.begin() + static_cast<std::ptrdiff_t>(from);
      const auto window_end = text_cp.begin() + static_cast<std::ptrdiff_t>(to);
      const auto match_it = std::search(window_begin, window_end, lookup_cp.begin(), lookup_cp.end());
      if (match_it != window_end) {
        result_index = static_cast<int64_t>(std::distance(text_cp.begin(), match_it));
      }
    }
    result_obj.SetValue(result_index);
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
  }
}
