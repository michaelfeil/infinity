// SPDX-License-Identifier: Apache-2.0
/// \file
/// Per-device accounting for retained native workspace allocations.
/// Live engine policies contribute limits and allocation permits return budget on release.

#pragma once
#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace tei {
/// Per-device retained-workspace accounting and contributions from live engine policies.
struct WorkspaceAdmissionState {
  std::mutex mutex;
  size_t limit = 0, retained = 0;
  std::multiset<size_t> policies;
};

inline std::shared_ptr<WorkspaceAdmissionState> workspace_admission_state(int index) {
  static std::mutex mutex;
  static std::unordered_map<int, std::shared_ptr<WorkspaceAdmissionState>> devices;
  std::lock_guard lock(mutex);
  auto& state = devices[index];
  if (!state)
    state = std::make_shared<WorkspaceAdmissionState>();
  return state;
}

/// Registers a live engine allowance in the shared device workspace budget.
class WorkspaceMemoryPolicy {
  std::shared_ptr<WorkspaceAdmissionState> state;
  std::multiset<size_t>::iterator contribution;

public:
  WorkspaceMemoryPolicy(std::shared_ptr<WorkspaceAdmissionState> owner, size_t limit)
      : state(std::move(owner)) {
    if (!limit)
      throw std::invalid_argument("Native retained workspace allowance must be positive");
    std::lock_guard lock(state->mutex);
    contribution = state->policies.insert(limit);
    state->limit = *state->policies.begin();
  }

  WorkspaceMemoryPolicy(const WorkspaceMemoryPolicy&) = delete;

  ~WorkspaceMemoryPolicy() {
    std::lock_guard lock(state->mutex);
    state->policies.erase(contribution);
    state->limit = state->policies.empty() ? 0 : *state->policies.begin();
  }
};

inline std::shared_ptr<WorkspaceMemoryPolicy> configure_workspace_admission(int index,
                                                                            size_t limit) {
  // Only live engines contribute to the most restrictive positive policy.
  return std::make_shared<WorkspaceMemoryPolicy>(workspace_admission_state(index), limit);
}

/// Returns reserved workspace bytes to the shared budget when its owner releases it.
class WorkspaceMemoryPermit {
  std::shared_ptr<WorkspaceAdmissionState> state;
  size_t bytes;

public:
  WorkspaceMemoryPermit(std::shared_ptr<WorkspaceAdmissionState> owner, size_t count)
      : state(std::move(owner)), bytes(count) {}

  WorkspaceMemoryPermit(const WorkspaceMemoryPermit&) = delete;

  ~WorkspaceMemoryPermit() {
    std::lock_guard lock(state->mutex);
    state->retained -= bytes;
  }
};

inline std::shared_ptr<WorkspaceMemoryPermit> reserve_workspace_memory(int index, size_t bytes) {
  auto state = workspace_admission_state(index);
  std::lock_guard lock(state->mutex);
  const auto limit = state->limit ? state->limit : size_t(512) << 20;
  if (state->retained > limit || bytes > limit - state->retained)
    return {};
  auto permit = std::make_shared<WorkspaceMemoryPermit>(state, bytes);
  state->retained += bytes;
  return permit;
}
}
