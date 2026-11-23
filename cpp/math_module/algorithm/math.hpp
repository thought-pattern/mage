#pragma once

#include <mgp.hpp>
#include <string>
#include <string_view>

namespace Math {

enum class RoundingMode : uint8_t { CEILING, FLOOR, UP, DOWN, HALF_EVEN, HALF_DOWN, HALF_UP, UNNECESSARY };

const std::string kProcedureRound = "round";
const std::string kArgumentValue = "value";
const std::string kArgumentPrecision = "precision";
const std::string kArgumentMode = "mode";
const std::string kArgumentResult = "result";

RoundingMode StringToRoundingMode(std::string_view mode_str);
RoundingMode StringToRoundingMode(const std::string &mode_str);

void Round(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory);

}  // namespace Math
