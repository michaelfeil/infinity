// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "helpers/admission_error.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
namespace tei {
inline size_t device_admission_available(size_t driver_free,size_t total,double fraction,int64_t reserved) {
  TORCH_CHECK(std::isfinite(fraction)&&fraction>=0&&fraction<=1&&reserved>=0,
    "Invalid native allocator quota accounting");
  const auto allowed=static_cast<size_t>(std::floor(static_cast<long double>(total)*fraction));
  const auto quota_free=uint64_t(reserved)<=allowed?allowed-static_cast<size_t>(reserved):size_t(0);
  return std::min(driver_free,quota_free);
}
struct DeviceAdmissionState { std::mutex mutex;size_t pending=0; };
inline std::shared_ptr<DeviceAdmissionState> device_admission_state(int index) {
  static std::mutex mutex;
  static std::unordered_map<int,std::shared_ptr<DeviceAdmissionState>> devices;
  std::lock_guard lock(mutex);
  auto& state=devices[index];if(!state)state=std::make_shared<DeviceAdmissionState>();return state;
}
class DeviceMemoryPermit {
  std::shared_ptr<DeviceAdmissionState> state;
  size_t bytes;
public:
  // Caller holds state->mutex across the physical free-memory query and reserve.
  DeviceMemoryPermit(std::shared_ptr<DeviceAdmissionState> owner,size_t free,size_t headroom,size_t request)
    :state(std::move(owner)),bytes(request) {
    require_memory_admission(headroom<=free&&state->pending<=free-headroom&&
      request<=free-headroom-state->pending,"Native device memory headroom is occupied");
    state->pending+=request;
  }
  DeviceMemoryPermit(const DeviceMemoryPermit&)=delete;
  DeviceMemoryPermit& operator=(const DeviceMemoryPermit&)=delete;
  ~DeviceMemoryPermit(){std::lock_guard lock(state->mutex);state->pending-=bytes;}
};
}
