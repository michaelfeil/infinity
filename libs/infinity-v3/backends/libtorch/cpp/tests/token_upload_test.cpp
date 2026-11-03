// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// Packed token upload ownership, packing and staging-budget regression tests.

#include "batching/output.h"
#include "batching/token_upload.h"
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace {
using namespace infinity;

struct Tokens {
  std::vector<int64_t> ids, types, positions;
  std::vector<int32_t> cumulative;

  int64_t count() const {
    return static_cast<int64_t>(ids.size());
  }

  int64_t batch() const {
    return static_cast<int64_t>(cumulative.size()) - 1;
  }
};

Tokens make_tokens(int64_t tokens, int64_t batch, int64_t seed = 0) {
  TORCH_CHECK(tokens >= batch && batch > 0, "Invalid test token geometry");
  Tokens value;
  value.ids.resize(tokens);
  value.types.resize(tokens);
  value.positions.resize(tokens);
  value.cumulative.assign(batch + 1, 0);
  for (int64_t i = 0; i < tokens; ++i) {
    value.ids[i] = seed + i + 3;
    value.types[i] = i % 2;
    value.positions[i] = i % 7;
  }
  const auto base = tokens / batch, extra = tokens % batch;
  int64_t offset = 0;
  for (int64_t row = 0; row < batch; ++row) {
    offset += base + (row < extra ? 1 : 0);
    value.cumulative[row + 1] = static_cast<int32_t>(offset);
  }
  return value;
}

template <class Call> void rejects(Call&& call, const char* message) {
  bool failed = false;
  try {
    call();
  } catch (const std::exception&) {
    failed = true;
  }
  TORCH_CHECK(failed, message);
}

void check_layout(const TokenUploadLayout& layout, int64_t tokens, int64_t batch) {
  TORCH_CHECK(layout.tokens == tokens && layout.batch == batch &&
                  layout.token_bytes == tokens * static_cast<int64_t>(sizeof(int64_t)) &&
                  layout.cumulative_bytes == (batch + 1) * static_cast<int64_t>(sizeof(int32_t)) &&
                  layout.total_bytes == 3 * layout.token_bytes + layout.cumulative_bytes,
              "Packed token layout changed typed extents");
}

void check_upload(const TokenUpload& uploaded, const Tokens& expected) {
  const auto layout = token_upload_layout(expected.count(), expected.batch());
  TORCH_CHECK(uploaded.storage.defined() && uploaded.storage.scalar_type() == at::kByte &&
                  uploaded.storage.dim() == 1 && uploaded.storage.numel() == layout.total_bytes,
              "Packed token storage has the wrong byte geometry");
  TORCH_CHECK(uploaded.ids.scalar_type() == at::kLong &&
                  uploaded.types.scalar_type() == at::kLong &&
                  uploaded.positions.scalar_type() == at::kLong &&
                  uploaded.cumulative.scalar_type() == at::kInt,
              "Packed token upload changed typed dtypes");
  TORCH_CHECK(uploaded.ids.sizes() == at::IntArrayRef({expected.count()}) &&
                  uploaded.types.sizes() == uploaded.ids.sizes() &&
                  uploaded.positions.sizes() == uploaded.ids.sizes() &&
                  uploaded.cumulative.sizes() == at::IntArrayRef({expected.batch() + 1}),
              "Packed token upload changed typed shapes");
  TORCH_CHECK(uploaded.ids.is_contiguous() && uploaded.types.is_contiguous() &&
                  uploaded.positions.is_contiguous() && uploaded.cumulative.is_contiguous(),
              "Packed token upload produced non-contiguous views");
  const auto storage = uploaded.storage.storage().data_ptr().get();
  TORCH_CHECK(uploaded.ids.storage().data_ptr().get() == storage &&
                  uploaded.types.storage().data_ptr().get() == storage &&
                  uploaded.positions.storage().data_ptr().get() == storage &&
                  uploaded.cumulative.storage().data_ptr().get() == storage,
              "Packed token views do not share their parent storage");
  const auto base = reinterpret_cast<uintptr_t>(uploaded.storage.const_data_ptr());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(uploaded.ids.const_data_ptr()) == base &&
                  reinterpret_cast<uintptr_t>(uploaded.types.const_data_ptr()) ==
                      base + static_cast<uintptr_t>(layout.token_bytes) &&
                  reinterpret_cast<uintptr_t>(uploaded.positions.const_data_ptr()) ==
                      base + static_cast<uintptr_t>(2 * layout.token_bytes) &&
                  reinterpret_cast<uintptr_t>(uploaded.cumulative.const_data_ptr()) ==
                      base + static_cast<uintptr_t>(3 * layout.token_bytes),
              "Packed token views are not placed at the packed offsets");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(uploaded.ids.const_data_ptr()) % alignof(int64_t) == 0 &&
          reinterpret_cast<uintptr_t>(uploaded.types.const_data_ptr()) % alignof(int64_t) == 0 &&
          reinterpret_cast<uintptr_t>(uploaded.positions.const_data_ptr()) % alignof(int64_t) ==
              0 &&
          reinterpret_cast<uintptr_t>(uploaded.cumulative.const_data_ptr()) % alignof(int32_t) == 0,
      "Packed token views are not naturally aligned");
  TORCH_CHECK(at::equal(uploaded.ids.cpu(),
                        at::tensor(expected.ids, at::TensorOptions().dtype(at::kLong))) &&
                  at::equal(uploaded.types.cpu(),
                            at::tensor(expected.types, at::TensorOptions().dtype(at::kLong))) &&
                  at::equal(uploaded.positions.cpu(),
                            at::tensor(expected.positions, at::TensorOptions().dtype(at::kLong))) &&
                  at::equal(uploaded.cumulative.cpu(),
                            at::tensor(expected.cumulative, at::TensorOptions().dtype(at::kInt))),
              "Packed token upload changed raw values or cumulative boundaries");
}

void layout_and_rejection_test() {
  check_layout(token_upload_layout(5, 2), 5, 2);
  check_layout(token_upload_layout(3, 2), 3, 2);
  check_layout(token_upload_layout(1, 1), 1, 1);
  rejects([] { (void)token_upload_layout(0, 1); }, "Zero tokens were accepted");
  rejects([] { (void)token_upload_layout(1, 0); }, "Zero batch was accepted");
  rejects([] { (void)token_upload_layout(std::numeric_limits<int64_t>::max(), 2); },
          "Overflowing token bytes were accepted");
  rejects([] { (void)token_upload_layout(5, std::numeric_limits<int64_t>::max()); },
          "Overflowing cumulative bytes were accepted");
  int64_t ids[] = {1, 2};
  int64_t types[] = {0, 0};
  int64_t positions[] = {0, 1};
  int32_t cumulative[] = {0, 2};
  rejects(
      [&] { (void)upload_tokens(nullptr, types, positions, cumulative, 2, 1, c10::Device("cpu")); },
      "Null IDs were accepted");
  rejects(
      [&] { (void)upload_tokens(ids, nullptr, positions, cumulative, 2, 1, c10::Device("cpu")); },
      "Null types were accepted");
  rejects([&] { (void)upload_tokens(ids, types, nullptr, cumulative, 2, 1, c10::Device("cpu")); },
          "Null positions were accepted");
  rejects([&] { (void)upload_tokens(ids, types, positions, nullptr, 2, 1, c10::Device("cpu")); },
          "Null cumulative offsets were accepted");
}

void raw_cpu_test() {
  Tokens value;
  value.ids = {std::numeric_limits<int64_t>::min(), -1, 0x0102030405060708LL, 42, -42};
  value.types = {0, 1, -1, 2, 3};
  value.positions = {5, 4, 3, 2, 1};
  value.cumulative = {0, 2, 5};
  auto uploaded = upload_tokens(value.ids.data(),
                                value.types.data(),
                                value.positions.data(),
                                value.cumulative.data(),
                                value.count(),
                                value.batch(),
                                c10::Device("cpu"));
  check_layout(token_upload_layout(value.count(), value.batch()), value.count(), value.batch());
  check_upload(uploaded, value);

  // A retained typed view must own the parent allocation after the wrapper dies.
  auto ids = uploaded.ids;
  auto types = uploaded.types;
  auto expected_ids = ids.cpu().clone();
  auto expected_types = types.cpu().clone();
  uploaded = TokenUpload{};
  TORCH_CHECK(at::equal(ids.cpu(), expected_ids) && at::equal(types.cpu(), expected_types),
              "Packed token views did not survive wrapper destruction");
}

void varying_geometry_cpu_test() {
  for (const auto& [tokens, batch] :
       std::vector<std::pair<int64_t, int64_t>>{{3, 1}, {17, 4}, {1, 1}, {9, 3}, {33, 6}, {2, 2}}) {
    auto value = make_tokens(tokens, batch, tokens);
    auto uploaded = upload_tokens(value.ids.data(),
                                  value.types.data(),
                                  value.positions.data(),
                                  value.cumulative.data(),
                                  tokens,
                                  batch,
                                  c10::Device("cpu"));
    check_upload(uploaded, value);
  }
}

std::shared_ptr<OutputPool> make_pool(size_t bytes) {
  auto pool = std::make_shared<OutputPool>();
  pool->max_bytes = bytes;
  return pool;
}

#ifdef TEI_TORCH_CUDA_KERNELS
void cuda_sync_test(const c10::Device& device) {
  auto value = make_tokens(5, 2, 7);
  auto uploaded = upload_tokens(value.ids.data(),
                                value.types.data(),
                                value.positions.data(),
                                value.cumulative.data(),
                                value.count(),
                                value.batch(),
                                device);
  check_upload(uploaded, value);

  auto cumulative = uploaded.cumulative;
  auto expected = cumulative.cpu().clone();
  uploaded = TokenUpload{};
  TORCH_CHECK(at::equal(cumulative.cpu(), expected),
              "CUDA packed token view did not survive wrapper destruction");
}

void cuda_async_ownership_test(const c10::Device& device) {
  auto pool = make_pool(1ULL << 20);
  OutputLease lease(pool, device, 8);
  lease.allocate();

  auto value = make_tokens(5, 2, 11);
  auto ids = value.ids;
  auto uploaded = upload_tokens(ids.data(),
                                value.types.data(),
                                value.positions.data(),
                                value.cumulative.data(),
                                value.count(),
                                value.batch(),
                                lease);
  auto ids_view = uploaded.ids;
  auto types_view = uploaded.types;
  auto positions_view = uploaded.positions;
  auto cumulative_view = uploaded.cumulative;
  auto sum = ids_view.to(at::kFloat).sum() + types_view.to(at::kFloat).sum() +
             positions_view.to(at::kFloat).sum() + cumulative_view.to(at::kFloat).sum();

  // The pin already owns the bytes; caller storage can be overwritten and released.
  std::fill(ids.begin(), ids.end(), 99);
  std::fill(value.types.begin(), value.types.end(), 99);
  std::fill(value.positions.begin(), value.positions.end(), 99);
  std::fill(value.cumulative.begin(), value.cumulative.end(), 99);
  value = Tokens{};
  uploaded = TokenUpload{};

  lease.record();
  lease.wait();

  auto expected = make_tokens(5, 2, 11);
  TORCH_CHECK(
      at::equal(ids_view.cpu(), at::tensor(expected.ids, at::TensorOptions().dtype(at::kLong))) &&
          at::equal(types_view.cpu(),
                    at::tensor(expected.types, at::TensorOptions().dtype(at::kLong))) &&
          at::equal(positions_view.cpu(),
                    at::tensor(expected.positions, at::TensorOptions().dtype(at::kLong))) &&
          at::equal(cumulative_view.cpu(),
                    at::tensor(expected.cumulative, at::TensorOptions().dtype(at::kInt))),
      "Async packed upload aliased mutable caller storage");
  float expected_sum = 0;
  for (int64_t index = 0; index < expected.count(); ++index) {
    expected_sum += static_cast<float>(expected.ids[index]);
    expected_sum += static_cast<float>(expected.types[index]);
    expected_sum += static_cast<float>(expected.positions[index]);
  }
  for (auto offset : expected.cumulative)
    expected_sum += static_cast<float>(offset);
  TORCH_CHECK(std::abs(sum.item<float>() - expected_sum) < 1e-6,
              "Async packed views changed before lease-stream kernels finished");
}

void cuda_async_slot_reuse_test(const c10::Device& device) {
  auto pool = make_pool(8ULL << 20);
  for (const auto& [tokens, batch] :
       std::vector<std::pair<int64_t, int64_t>>{{3, 1}, {17, 4}, {1, 1}, {9, 3}, {33, 6}, {2, 2}}) {
    auto value = make_tokens(tokens, batch, tokens);
    OutputLease lease(pool, device, 4);
    lease.allocate();
    auto uploaded = upload_tokens(value.ids.data(),
                                  value.types.data(),
                                  value.positions.data(),
                                  value.cumulative.data(),
                                  tokens,
                                  batch,
                                  lease);
    lease.record();
    lease.wait();
    check_upload(uploaded, value);
  }
}

void cuda_async_budget_recovery_test(const c10::Device& device) {
  auto pool = make_pool(8192);
  auto pending = std::make_unique<OutputLease>(pool, device, 1);
  pending->allocate();
  pending->record();
  {
    OutputLease blocked(pool, device, 1);
    blocked.allocate();
    auto value = make_tokens(340, 4, 3);
    rejects(
        [&] {
          (void)upload_tokens(value.ids.data(),
                              value.types.data(),
                              value.positions.data(),
                              value.cumulative.data(),
                              value.count(),
                              value.batch(),
                              blocked);
        },
        "Packed upload bypassed the occupied output-staging budget");
  }
  pending.reset();

  auto value = make_tokens(340, 4, 3);
  OutputLease recovered(pool, device, 1);
  recovered.allocate();
  auto uploaded = upload_tokens(value.ids.data(),
                                value.types.data(),
                                value.positions.data(),
                                value.cumulative.data(),
                                value.count(),
                                value.batch(),
                                recovered);
  recovered.record();
  recovered.wait();
  check_upload(uploaded, value);
}
#endif
}

int main(int argc, char** argv) {
  c10::InferenceMode inference;
  layout_and_rejection_test();
  raw_cpu_test();
  varying_geometry_cpu_test();
#ifdef TEI_TORCH_CUDA_KERNELS
  if (argc > 1 && std::string(argv[1]) == "--cuda") {
    const c10::Device device("cuda:0");
    cuda_sync_test(device);
    cuda_async_ownership_test(device);
    cuda_async_slot_reuse_test(device);
    cuda_async_budget_recovery_test(device);
  } else {
    TORCH_CHECK(argc == 1, "Unknown token_upload_test argument");
  }
#else
  TORCH_CHECK(argc == 1, "CUDA token upload tests require a CUDA build");
#endif
  std::cout << "token upload packing, ownership and staging budget tests passed\n";
}
