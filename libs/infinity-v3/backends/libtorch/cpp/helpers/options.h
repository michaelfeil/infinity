/// \file
/// Configuration accessors and named weight lookup shared by native model families.
/// Required weights are checked by name; option accessors provide typed reads
/// with defaults for missing keys.

#pragma once
#include <ATen/ATen.h>
#include <string>
#include <unordered_map>

namespace tei {
using Weights = std::unordered_map<std::string, at::Tensor>;

/// String-valued model configuration with typed lookups and missing-key defaults.
/// Present numeric values are parsed with the standard string conversion functions.
struct Options {
  std::unordered_map<std::string, std::string> values;

  std::string string(const std::string& key, const std::string& fallback = "") const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }

  int64_t integer(const std::string& key, int64_t fallback = 0) const {
    auto it = values.find(key);
    return it == values.end() ? fallback : std::stoll(it->second);
  }

  double real(const std::string& key, double fallback = 0) const {
    auto it = values.find(key);
    return it == values.end() ? fallback : std::stod(it->second);
  }

  bool boolean(const std::string& key, bool fallback = false) const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second == "true";
  }
};

inline const at::Tensor& weight(const Weights& weights, const std::string& name) {
  auto it = weights.find(name);
  TORCH_CHECK(it != weights.end(), "Missing model weight: ", name);
  return it->second;
}
}
