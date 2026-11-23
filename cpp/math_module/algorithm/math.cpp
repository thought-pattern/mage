#include "math.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Math {
namespace {

double RoundUnitValue(double scaled_value, double original_sign_source, RoundingMode mode) {
  switch (mode) {
    case RoundingMode::CEILING:
      return std::ceil(scaled_value);

    case RoundingMode::FLOOR:
      return std::floor(scaled_value);

    case RoundingMode::UP:
      return original_sign_source >= 0.0 ? std::ceil(scaled_value) : std::floor(scaled_value);

    case RoundingMode::DOWN:
      return original_sign_source >= 0.0 ? std::floor(scaled_value) : std::ceil(scaled_value);

    case RoundingMode::HALF_UP:
      return std::round(scaled_value);

    case RoundingMode::HALF_DOWN: {
      const double abs_val = std::abs(scaled_value);
      const double floor_abs = std::floor(abs_val);
      const double frac_abs = abs_val - floor_abs;
      double rounded_abs = floor_abs;
      if (frac_abs > 0.5) {
        rounded_abs = std::ceil(abs_val);
      }
      return scaled_value < 0.0 ? -rounded_abs : rounded_abs;
    }

    case RoundingMode::HALF_EVEN: {
      const double floor_val = std::floor(scaled_value);
      const double fractional_part = scaled_value - floor_val;
      if (fractional_part < 0.5) {
        return floor_val;
      }
      if (fractional_part > 0.5) {
        return std::ceil(scaled_value);
      }
      const double half_floor = std::floor(floor_val * 0.5);
      if (half_floor * 2.0 == floor_val) {
        return floor_val;
      }
      return std::ceil(scaled_value);
    }

    case RoundingMode::UNNECESSARY: {
      if (scaled_value == std::floor(scaled_value)) {
        return scaled_value;
      }
      throw std::runtime_error("Rounding necessary for UNNECESSARY mode");
    }

    default:
      throw std::invalid_argument("Unknown rounding mode");
  }
}

}  // namespace

RoundingMode StringToRoundingMode(std::string_view mode_str) {
  if (mode_str == "CEILING") return RoundingMode::CEILING;
  if (mode_str == "FLOOR") return RoundingMode::FLOOR;
  if (mode_str == "UP") return RoundingMode::UP;
  if (mode_str == "DOWN") return RoundingMode::DOWN;
  if (mode_str == "HALF_EVEN") return RoundingMode::HALF_EVEN;
  if (mode_str == "HALF_DOWN") return RoundingMode::HALF_DOWN;
  if (mode_str == "HALF_UP") return RoundingMode::HALF_UP;
  if (mode_str == "UNNECESSARY") return RoundingMode::UNNECESSARY;
  throw std::invalid_argument("Invalid rounding mode: " + std::string(mode_str));
}

RoundingMode StringToRoundingMode(const std::string &mode_str) {
  return StringToRoundingMode(std::string_view(mode_str));
}

double ApplyRounding(double value, std::int64_t precision, RoundingMode mode) {
  if (!std::isfinite(value) || value == 0.0) {
    return value;
  }

  constexpr double kExactIntBound = 9007199254740992.0;  // 2^53

  if (precision >= 0) {
    if (precision >= 1075 || std::abs(value) >= kExactIntBound) {
      return value;
    }
    if (precision <= 290) {
      const double multiplier = std::pow(10.0, static_cast<double>(precision));
      const double scaled_value = value * multiplier;
      if (!std::isfinite(scaled_value) || std::abs(scaled_value) >= kExactIntBound) {
        return value;
      }
      return RoundUnitValue(scaled_value, value, mode) / multiplier;
    }

    // Two-step scaling for 291 <= precision <= 1074 on sub-10^-290 magnitudes
    double working = value;
    std::int64_t rem = precision;
    int chunk_count = 0;
    while (rem > 200) {
      working *= 1e200;
      rem -= 200;
      ++chunk_count;
      if (!std::isfinite(working) || std::abs(working) >= kExactIntBound) {
        return value;
      }
    }
    const double tail_mult = std::pow(10.0, static_cast<double>(rem));
    const double scaled = working * tail_mult;
    if (!std::isfinite(scaled) || std::abs(scaled) >= kExactIntBound) {
      return value;
    }
    double rounded = RoundUnitValue(scaled, value, mode) / tail_mult;
    for (int i = 0; i < chunk_count; ++i) {
      rounded /= 1e200;
    }
    return rounded;
  }

  // Negative precision: rounding to multiples of 10^(-precision)
  if (precision <= -309) {
    // Every finite double satisfies |value| < 1.8e308 < 0.5 * 10^309
    switch (mode) {
      case RoundingMode::CEILING:
        return value > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
      case RoundingMode::FLOOR:
        return value < 0.0 ? -std::numeric_limits<double>::infinity() : 0.0;
      case RoundingMode::UP:
        return value > 0.0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
      case RoundingMode::DOWN:
      case RoundingMode::HALF_UP:
      case RoundingMode::HALF_DOWN:
      case RoundingMode::HALF_EVEN:
        return 0.0;
      case RoundingMode::UNNECESSARY:
        throw std::runtime_error("Rounding necessary for UNNECESSARY mode");
      default:
        throw std::invalid_argument("Unknown rounding mode");
    }
  }

  const double divisor = std::pow(10.0, static_cast<double>(-precision));
  const double scaled_value = value / divisor;
  return RoundUnitValue(scaled_value, value, mode) * divisor;
}

void Round(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  (void)ctx;
  mgp::MemoryDispatcherGuard guard{memory};
  auto result = mgp::Result(res);
  const auto arguments = mgp::List(args);

  try {
    const auto value_arg = arguments[0];
    const auto precision_arg = arguments[1];
    const auto mode_arg = arguments[2];

    // invalid type, return null
    if (!value_arg.IsDouble() || !precision_arg.IsInt() || !mode_arg.IsString()) {
      result.SetValue();
      return;
    }

    auto value = value_arg.ValueDouble();
    auto precision = precision_arg.ValueInt();
    auto mode_str = mode_arg.ValueString();

    RoundingMode mode = StringToRoundingMode(mode_str);
    double rounded_value = ApplyRounding(value, precision, mode);

    result.SetValue(rounded_value);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
  }
}

}  // namespace Math
