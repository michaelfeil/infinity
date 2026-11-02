// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Torch-free provider boundary/GQA/stride/local-mask and sanitizer test.
#include "head512_provider.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kD = 512;

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "head512 provider test FAIL: %s\n", message.c_str());
  std::exit(1);
}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    fail(std::string(what) + ": " + cudaGetErrorString(status));
}

struct Device {
  void* ptr = nullptr;
  size_t bytes = 0;
  Device() = default;
  explicit Device(size_t n) : bytes(n) { check(cudaMalloc(&ptr, n), "cudaMalloc"); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { if (ptr) cudaFree(ptr); }
};

enum class DType { F16, Bf16 };
enum class Window { Global, Local };

struct Case {
  const char* name;
  DType dtype;
  bool causal;
  Window window;
  int left;
  int right;
  std::vector<int> q_lengths;
  std::vector<int> k_lengths;
  int heads;
  int kv_heads;
};

uint16_t random_bits(DType dtype, uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  const float value = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.8f;
  if (dtype == DType::F16) {
    __half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  }
  __nv_bfloat16 b = __float2bfloat16_rn(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &b, sizeof(bits));
  return bits;
}

float decode(DType dtype, uint16_t bits) {
  if (dtype == DType::F16) {
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
  }
  __nv_bfloat16 b;
  std::memcpy(&b, &bits, sizeof(bits));
  return __bfloat162float(b);
}

bool allowed(const Case& c, int query, int key) {
  if (c.causal && key > query)
    return false;
  if (c.window == Window::Local) {
    if (query - key > c.left)
      return false;
    if (key - query > c.right)
      return false;
  }
  return true;
}

struct Layout {
  int q_head = kD + 32;
  int q_row = 0;
  int kv_head = kD + 24;
  int kv_row = 0;
  int o_head = kD + 16;
  int o_row = 0;
  size_t q_elems = 0;
  size_t kv_elems = 0;
  size_t o_elems = 0;
};

Layout make_layout(const Case& c, int total_q, int total_k) {
  Layout l;
  l.q_row = c.heads * l.q_head + 16;
  l.kv_row = c.kv_heads * l.kv_head + 8;
  l.o_row = c.heads * l.o_head + 8;
  l.q_elems = static_cast<size_t>(total_q - 1) * l.q_row + static_cast<size_t>(c.heads - 1) * l.q_head + kD;
  l.kv_elems = static_cast<size_t>(total_k - 1) * l.kv_row + static_cast<size_t>(c.kv_heads - 1) * l.kv_head + kD;
  l.o_elems = static_cast<size_t>(total_q - 1) * l.o_row + static_cast<size_t>(c.heads - 1) * l.o_head + kD;
  return l;
}

double at_logical(const std::vector<uint16_t>& data, DType dtype, int token, int head, int dim,
                  int row_stride, int head_stride) {
  return decode(dtype, data[static_cast<size_t>(token) * row_stride + static_cast<size_t>(head) * head_stride + dim]);
}

struct CpuRef {
  std::vector<double> out;
  std::vector<double> lse;
};

CpuRef reference(const Case& c, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
                 const std::vector<uint16_t>& v, const Layout& l,
                 const std::vector<int32_t>& q_offsets, const std::vector<int32_t>& k_offsets,
                 int total_q, int total_k, float scale) {
  CpuRef r;
  r.out.assign(static_cast<size_t>(total_q) * c.heads * kD, 0.0);
  r.lse.assign(static_cast<size_t>(total_q) * c.heads, 0.0);
  const int group = c.heads / c.kv_heads;
  std::vector<double> scores;
  for (int b = 0; b < static_cast<int>(q_offsets.size()) - 1; ++b) {
    const int q0 = q_offsets[b], q1 = q_offsets[b + 1];
    const int k0 = k_offsets[b], k1 = k_offsets[b + 1];
    const int n = q1 - q0, m = k1 - k0;
    scores.resize(static_cast<size_t>(m));
    for (int h = 0; h < c.heads; ++h) {
      const int kh = h / group;
      for (int i = 0; i < n; ++i) {
        double maximum = -std::numeric_limits<double>::infinity();
        int used = 0;
        for (int j = 0; j < m; ++j) {
          if (!allowed(c, i, j)) {
            scores[j] = -std::numeric_limits<double>::infinity();
            continue;
          }
          double dot = 0.0;
          for (int d = 0; d < kD; ++d)
            dot += at_logical(q, c.dtype, q0 + i, h, d, l.q_row, l.q_head) *
                   at_logical(k, c.dtype, k0 + j, kh, d, l.kv_row, l.kv_head);
          scores[j] = dot * scale;
          maximum = std::max(maximum, scores[j]);
          ++used;
        }
        if (used == 0)
          fail("reference produced an empty attention row");
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
          if (scores[j] == -std::numeric_limits<double>::infinity())
            continue;
          scores[j] = std::exp(scores[j] - maximum);
          sum += scores[j];
        }
        const double logsum = std::log(sum) + maximum;
        r.lse[static_cast<size_t>(h) * total_q + (q0 + i)] = logsum;
        for (int d = 0; d < kD; ++d) {
          double value = 0.0;
          for (int j = 0; j < m; ++j) {
            if (scores[j] == 0.0)
              continue;
            value += scores[j] * at_logical(v, c.dtype, k0 + j, kh, d, l.kv_row, l.kv_head);
          }
          r.out[(static_cast<size_t>(q0 + i) * c.heads + h) * kD + d] = value / sum;
        }
      }
    }
  }
  return r;
}

void run_case(const Case& c) {
  const int total_q = [&] { int n = 0; for (int x : c.q_lengths) n += x; return n; }();
  const int total_k = [&] { int n = 0; for (int x : c.k_lengths) n += x; return n; }();
  const int max_q = *std::max_element(c.q_lengths.begin(), c.q_lengths.end());
  const int max_k = *std::max_element(c.k_lengths.begin(), c.k_lengths.end());
  if (c.k_lengths.size() != c.q_lengths.size())
    fail("test q/k batch mismatch");
  std::vector<int32_t> q_offsets{0}, k_offsets{0};
  for (int n : c.q_lengths) q_offsets.push_back(q_offsets.back() + n);
  for (int n : c.k_lengths) k_offsets.push_back(k_offsets.back() + n);
  const Layout l = make_layout(c, total_q, total_k);
  uint32_t state = 0x12345678u;
  std::vector<uint16_t> q(l.q_elems), k(l.kv_elems), v(l.kv_elems);
  for (auto& value : q) value = random_bits(c.dtype, state);
  for (auto& value : k) value = random_bits(c.dtype, state);
  for (auto& value : v) value = random_bits(c.dtype, state);
  std::vector<uint16_t> out(l.o_elems, 0);
  std::vector<float> lse(static_cast<size_t>(total_q) * c.heads, -7.0f);

  Device dq(q.size() * sizeof(uint16_t)), dk(k.size() * sizeof(uint16_t)),
      dv(v.size() * sizeof(uint16_t)), dqo(q_offsets.size() * sizeof(int32_t)),
      dko(k_offsets.size() * sizeof(int32_t)), dout(out.size() * sizeof(uint16_t)),
      dlse(lse.size() * sizeof(float));
  check(cudaMemcpy(dq.ptr, q.data(), dq.bytes, cudaMemcpyHostToDevice), "copy q");
  check(cudaMemcpy(dk.ptr, k.data(), dk.bytes, cudaMemcpyHostToDevice), "copy k");
  check(cudaMemcpy(dv.ptr, v.data(), dv.bytes, cudaMemcpyHostToDevice), "copy v");
  check(cudaMemcpy(dqo.ptr, q_offsets.data(), dqo.bytes, cudaMemcpyHostToDevice), "copy q offsets");
  check(cudaMemcpy(dko.ptr, k_offsets.data(), dko.bytes, cudaMemcpyHostToDevice), "copy k offsets");

  InfinityHead512Info info{};
  if (infinity_head512_info_v2(&info) != 0 || info.head_dim != kD || info.dtype_mask != 3 ||
      info.window_mask != 3 || info.dynamic_shared_memory_bytes != INFINITY_HEAD512_DYNAMIC_SMEM_BYTES ||
      info.min_optin_shared_memory_bytes != INFINITY_HEAD512_DYNAMIC_SMEM_BYTES || info.flags != 0x0f)
    fail("info ABI mismatch");

  if (infinity_head512_prepare_v2() != 0)
    fail(std::string("cold prepare failed: ") + infinity_head512_error_v2());

  InfinityHead512Batch batch{};
  batch.q = reinterpret_cast<uint64_t>(dq.ptr);
  batch.k = reinterpret_cast<uint64_t>(dk.ptr);
  batch.v = reinterpret_cast<uint64_t>(dv.ptr);
  batch.q_offsets = reinterpret_cast<uint64_t>(dqo.ptr);
  batch.k_offsets = reinterpret_cast<uint64_t>(dko.ptr);
  batch.total_q = total_q;
  batch.total_k = total_k;
  batch.batch = static_cast<int64_t>(c.q_lengths.size());
  batch.max_q = max_q;
  batch.max_k = max_k;
  batch.heads = c.heads;
  batch.kv_heads = c.kv_heads;
  batch.scale = 1.0;
  batch.strides[0] = l.q_row; batch.strides[1] = l.q_head; batch.strides[2] = 1;
  batch.strides[3] = l.kv_row; batch.strides[4] = l.kv_head; batch.strides[5] = 1;
  batch.strides[6] = l.kv_row; batch.strides[7] = l.kv_head; batch.strides[8] = 1;
  batch.strides[9] = l.o_row; batch.strides[10] = l.o_head; batch.strides[11] = 1;
  batch.dtype = c.dtype == DType::F16 ? INFINITY_HEAD512_DTYPE_FP16 : INFINITY_HEAD512_DTYPE_BF16;
  batch.causal = c.causal ? 1 : 0;
  const bool local = c.window == Window::Local;
  batch.window_left = local ? c.left : -1;
  batch.window_right = local ? c.right : -1;

  if (infinity_head512_enqueue_v2(&batch, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) != 0)
    fail(std::string("enqueue failed: ") + infinity_head512_error_v2());
  check(cudaDeviceSynchronize(), "sync");
  check(cudaMemcpy(out.data(), dout.ptr, dout.bytes, cudaMemcpyDeviceToHost), "copy out");
  check(cudaMemcpy(lse.data(), dlse.ptr, dlse.bytes, cudaMemcpyDeviceToHost), "copy lse");

  const CpuRef expected = reference(c, q, k, v, l, q_offsets, k_offsets, total_q, total_k, 1.0f);
  double worst_output = 0.0, worst_lse = 0.0;
  for (int t = 0; t < total_q; ++t) {
    for (int h = 0; h < c.heads; ++h) {
      const double got_lse = lse[static_cast<size_t>(h) * total_q + t];
      const double want_lse = expected.lse[static_cast<size_t>(h) * total_q + t];
      if (!std::isfinite(got_lse))
        fail("provider produced non-finite lse");
      worst_lse = std::max(worst_lse, std::abs(got_lse - want_lse));
      for (int d = 0; d < kD; ++d) {
        const double got = decode(c.dtype, out[static_cast<size_t>(t) * l.o_row +
                                               static_cast<size_t>(h) * l.o_head + d]);
        const double want = expected.out[(static_cast<size_t>(t) * c.heads + h) * kD + d];
        if (!std::isfinite(got))
          fail("provider produced non-finite output");
        worst_output = std::max(worst_output, std::abs(got - want));
      }
    }
  }
  if (worst_output > 0.004)
    fail(std::string(c.name) + " output mismatch " + std::to_string(worst_output));
  if (worst_lse > 0.03)
    fail(std::string(c.name) + " lse mismatch " + std::to_string(worst_lse));

  InfinityHead512Batch invalid = batch;
  invalid.strides[2] = 2;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted d stride 2");
  invalid = batch;
  invalid.kv_heads = c.heads + 1;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted non-divisible GQA heads");
  invalid = batch;
  invalid.scale = 0.0;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted zero scale");
  invalid = batch;
  invalid.window_left = -2;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted window -2");
  if (infinity_head512_enqueue_v2(&batch, dout.ptr, dout.bytes - 1, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted short output capacity");
  const uint64_t dense_out_bytes = static_cast<uint64_t>(total_q) * c.heads * kD * 2;
  if (dense_out_bytes < dout.bytes &&
      infinity_head512_enqueue_v2(&batch, dout.ptr, dense_out_bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted dense capacity for padded output");
  invalid = batch;
  invalid.strides[9] = l.o_head;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted overlapping output rows");
  invalid = batch;
  invalid.q = batch.q + 2;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted misaligned q storage");
  invalid = batch;
  invalid.strides[4] = l.kv_head + 1;
  if (infinity_head512_enqueue_v2(&invalid, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted non-vector-aligned k head stride");
  if (infinity_head512_enqueue_v2(&batch, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes - 1, nullptr) == 0)
    fail("accepted short lse capacity");
  if (infinity_head512_enqueue_v2(nullptr, dout.ptr, dout.bytes, dlse.ptr, dlse.bytes, nullptr) == 0)
    fail("accepted null batch");

  std::printf("PASS %-28s dtype=%-4s causal=%d local=%d heads=%d/%d total_q=%d total_k=%d "
              "out_max=%.6g lse_max=%.6g\n",
              c.name, c.dtype == DType::F16 ? "fp16" : "bf16", int(c.causal), int(local),
              c.heads, c.kv_heads, total_q, total_k, worst_output, worst_lse);
}
}  // namespace

int main() {
  const std::vector<Case> cases = {
      {"same-small-global", DType::F16, false, Window::Global, 0, 0, {3, 17, 9}, {3, 17, 9}, 4, 4},
      {"same-small-causal-gqa", DType::Bf16, true, Window::Global, 0, 0, {3, 17, 9}, {3, 17, 9}, 4, 1},
      {"same-boundary-gqa", DType::F16, false, Window::Global, 0, 0, {32, 129, 17}, {32, 129, 17}, 4, 1},
      {"same-boundary-causal", DType::Bf16, true, Window::Global, 0, 0, {32, 129, 17}, {32, 129, 17}, 4, 4},
      {"separate-ragged-gqa", DType::Bf16, true, Window::Global, 0, 0, {3, 17, 9}, {5, 2, 13}, 8, 2},
      {"separate-ragged-noncausal", DType::F16, false, Window::Global, 0, 0, {3, 17, 9}, {5, 2, 13}, 8, 2},
      {"tile-edges", DType::Bf16, false, Window::Global, 0, 0, {1, 31, 32, 33, 63, 64, 65, 127, 128, 129}, {1, 31, 32, 33, 63, 64, 65, 127, 128, 129}, 4, 2},
      {"local-causal-left64", DType::F16, true, Window::Local, 64, 0, {65, 129}, {65, 129}, 4, 2},
      {"local-noncausal-64", DType::Bf16, false, Window::Local, 64, 64, {65, 129}, {65, 129}, 4, 2},
  };
  for (const Case& c : cases)
    run_case(c);
  std::printf("head512 provider test PASS cases=%zu\n", cases.size());
  return 0;
}
