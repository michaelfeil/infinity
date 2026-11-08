/// \file
/// Configuration accessors and named weight lookup shared by native model families.
/// Required weights are checked by name; option accessors provide typed reads
/// with defaults for missing keys.

#pragma once
#include <ATen/ATen.h>
#include <string>
#include <unordered_map>

namespace infinity {
using Weights = std::unordered_map<std::string, at::Tensor>;

/// Per-model projection quantization selected by the serving layer.
enum class Quantization { None, Fp8, Mxfp8, Nvfp4 };
/// Which eligible projections a quantization format applies to.
enum class QuantizationScope { Mlp, All };

inline const char* quantization_name(Quantization format) {
  switch (format) {
  case Quantization::None:
    return "none";
  case Quantization::Fp8:
    return "fp8";
  case Quantization::Mxfp8:
    return "mxfp8";
  case Quantization::Nvfp4:
    return "nvfp4";
  }
  return "unknown";
}

inline Quantization quantization_from_string(const std::string& value) {
  if (value == "none")
    return Quantization::None;
  if (value == "fp8")
    return Quantization::Fp8;
  if (value == "mxfp8")
    return Quantization::Mxfp8;
  if (value == "nvfp4")
    return Quantization::Nvfp4;
  TORCH_CHECK(
      false, "Unsupported quantization format '", value, "'; expected none, fp8, mxfp8, or nvfp4");
}

inline QuantizationScope quantization_scope_from_string(const std::string& value) {
  if (value == "mlp")
    return QuantizationScope::Mlp;
  if (value == "all")
    return QuantizationScope::All;
  TORCH_CHECK(false, "Unsupported quantization scope '", value, "'; expected mlp or all");
}

/// String-valued model configuration with typed lookups and missing-key defaults.
/// Present numeric values are parsed with the standard string conversion functions.
struct Options {
  std::unordered_map<std::string, std::string> values;

  std::string string(const std::string& key, const std::string& fallback = "") const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }

  /// Parse a present value with stoll; only a missing key uses the fallback.
  int64_t integer(const std::string& key, int64_t fallback = 0) const {
    auto it = values.find(key);
    return it == values.end() ? fallback : std::stoll(it->second);
  }

  /// Parse a present value with stod; only a missing key uses the fallback.
  double real(const std::string& key, double fallback = 0) const {
    auto it = values.find(key);
    return it == values.end() ? fallback : std::stod(it->second);
  }

  /// A present value is true only for the literal "true"; a missing key uses the fallback.
  bool boolean(const std::string& key, bool fallback = false) const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second == "true";
  }

  /// Projection quantization for this model; a missing key preserves model dtype.
  Quantization dynamic_quantization() const {
    auto it = values.find("_dynamic_quantization");
    return it == values.end() ? Quantization::None : quantization_from_string(it->second);
  }

  /// Resolved projection scope; the caller supplies the family-specific default.
  QuantizationScope
  dynamic_quantization_scope(QuantizationScope fallback = QuantizationScope::Mlp) const {
    auto it = values.find("_dynamic_quantization_scope");
    return it == values.end() ? fallback : quantization_scope_from_string(it->second);
  }
};

/// Borrow a required named tensor from the map; missing parameters raise a descriptive error.
inline const at::Tensor& weight(const Weights& weights, const std::string& name) {
  auto it = weights.find(name);
  TORCH_CHECK(it != weights.end(), "Missing model weight: ", name);
  return it->second;
}
}
