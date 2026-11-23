#include "csv_utils.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace CsvUtils {

fs::path DefaultCsvExportRoot() {
  std::error_code ec;
  const fs::path var_lib("/var/lib/memgraph");
  if (fs::exists(var_lib, ec) && fs::is_directory(var_lib, ec)) {
    const fs::path candidate = var_lib / "csv_exports";
    fs::create_directories(candidate, ec);
    if (!ec) {
      return candidate;
    }
  }
  const fs::path local_candidate = fs::current_path() / "csv_exports";
  fs::create_directories(local_candidate, ec);
  if (ec) {
    throw std::runtime_error("Failed to initialize CSV export directory.");
  }
  return local_candidate;
}

fs::path ResolveConfinedCsvPath(std::string_view raw_filepath, const fs::path &export_root) {
  if (raw_filepath.empty()) {
    throw std::invalid_argument("CSV filepath must not be empty.");
  }
  if (raw_filepath.find('\0') != std::string_view::npos) {
    throw std::invalid_argument("CSV filepath contains an invalid null byte.");
  }
  std::error_code ec;
  fs::create_directories(export_root, ec);
  if (ec) {
    throw std::runtime_error("CSV export root directory is unavailable.");
  }
  const fs::path canonical_root = fs::canonical(export_root);

  const fs::path requested(raw_filepath);
  for (const auto &part : requested) {
    if (part == "..") {
      throw std::invalid_argument("CSV filepath must not traverse parent directories.");
    }
  }
  if (requested.extension() != ".csv") {
    throw std::invalid_argument("CSV filepath must end with the '.csv' extension.");
  }

  fs::path relative_tail;
  if (!requested.is_absolute()) {
    relative_tail = requested.lexically_normal();
  } else {
    std::vector<fs::path> parts;
    bool matched_root = false;
    for (fs::path cur = requested; !cur.empty() && cur != cur.root_path(); cur = cur.parent_path()) {
      const auto cur_st = fs::symlink_status(cur, ec);
      if (!ec && fs::exists(cur_st) && !fs::is_symlink(cur_st)) {
        const fs::path cur_canon = fs::canonical(cur, ec);
        if (!ec && cur_canon == canonical_root) {
          matched_root = true;
          break;
        }
      }
      parts.push_back(cur.filename());
    }
    if (!matched_root || parts.empty()) {
      throw std::invalid_argument("CSV filepath must stay inside the designated CSV export directory.");
    }
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
      relative_tail /= *it;
    }
  }

  if (relative_tail.empty() || relative_tail == ".") {
    throw std::invalid_argument("CSV filepath must name a file inside the designated CSV export directory.");
  }

  fs::path check_path = canonical_root;
  for (const auto &part : relative_tail) {
    if (part.empty() || part == ".") {
      continue;
    }
    if (part == "..") {
      throw std::invalid_argument("CSV filepath must not traverse parent directories.");
    }
    check_path /= part;
    const auto st = fs::symlink_status(check_path, ec);
    if (!ec && fs::is_symlink(st)) {
      throw std::invalid_argument("CSV filepath must not traverse symbolic links.");
    }
  }
  return check_path;
}

void CreateCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto filepath = arguments[0].ValueString();
    const auto content = arguments[1].ValueString();
    const auto is_append = arguments[2].ValueBool();

    const fs::path resolved_path = ResolveConfinedCsvPath(filepath);
    if (!is_append && fs::exists(fs::symlink_status(resolved_path))) {
      throw std::runtime_error("CSV file already exists; set is_append=true or delete the existing export first.");
    }

    std::ofstream fout;
    fout.open(resolved_path, is_append ? std::ofstream::app : std::ofstream::out);
    if (!fout.is_open() || fout.fail()) {
      throw std::runtime_error("Failed to open CSV file '" + resolved_path.string() + "': " + std::strerror(errno));
    }
    fout << content;
    fout.flush();
    if (!fout) {
      throw std::runtime_error("Failed to write CSV file '" + resolved_path.string() + "': " + std::strerror(errno));
    }
    fout.close();
    if (fout.fail()) {
      throw std::runtime_error("Failed to close CSV file '" + resolved_path.string() + "': " + std::strerror(errno));
    }

    auto record = record_factory.NewRecord();
    const auto resolved_str = resolved_path.string();
    record.Insert(std::string(kArgumentCreateCsvFile1).c_str(), resolved_str);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
}

void DeleteCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string_view filepath = arguments[0].ValueString();
    const fs::path resolved_path = ResolveConfinedCsvPath(filepath);
    const auto status = fs::symlink_status(resolved_path);
    if (!fs::is_regular_file(status)) {
      record_factory.SetErrorMessage("Target CSV export file does not exist or is not a regular file");
      return;
    }
    if (!fs::remove(resolved_path)) {
      record_factory.SetErrorMessage("Failed to delete CSV file");
      return;
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
}

}  // namespace CsvUtils
