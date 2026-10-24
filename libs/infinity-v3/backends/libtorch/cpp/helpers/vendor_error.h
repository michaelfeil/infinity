// SPDX-License-Identifier: Apache-2.0
/// \file
/// Allocation-failure carrier for vendor operations with explicit status codes.

#pragma once
#include <cstdint>
#include <exception>

namespace tei {
// Only vendor calls that expose an actual allocation-failure status use this
// signal. Operation names have static storage; unwinding needs no message copy.
struct VendorAllocationError final : std::exception {
  const char* operation;
  int32_t status;

  VendorAllocationError(const char* operation_name, int32_t status_code) noexcept
      : operation(operation_name), status(status_code) {}

  const char* what() const noexcept override {
    return operation;
  }
};
}
