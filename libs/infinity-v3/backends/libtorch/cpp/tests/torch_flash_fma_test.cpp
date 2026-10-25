// Independent exact-length reference for the private native Torch Flash adapter.
#include "kernels/torch_flash_fma.h"
#include "kernels/torch_flash_exact32.h"
#include "batching/prefix_cache.h"
#include <ATen/ops/_cudnn_attention_forward.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/Parallel.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

// Isolate attention from shape-dependent projection rounding by supplying
// identical Q/K/V operands to complete and resumed causal execution.
static void cached_suffix_spans(const c10::Device& device) {
  const int64_t total = 512, heads = 4, kvheads = 2, dim = 128;
  auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto combined = at::randn({total, heads + 2 * kvheads, dim}, options);
  auto q = combined.narrow(1, 0, heads), k = combined.narrow(1, heads, kvheads),
       v = combined.narrow(1, heads + kvheads, kvheads);
  auto cumulative = at::tensor(std::vector<int32_t>{0, total}, options.dtype(at::kInt));
  auto reference =
      infinity::torch_flash_fma(q, k, v, cumulative, total, 1. / std::sqrt(double(dim)), true);
  for (int64_t prefix : {1, 13, 127, 129, 383, 511}) {
    auto count = total - prefix;
    auto suffix_cu = at::tensor(std::vector<int32_t>{0, int32_t(count)}, options.dtype(at::kInt));
    auto direct = infinity::torch_flash_fma_varlen(q.narrow(0, prefix, count),
                                                   k,
                                                   v,
                                                   suffix_cu,
                                                   cumulative,
                                                   count,
                                                   total,
                                                   1. / std::sqrt(double(dim)),
                                                   true);
    TORCH_CHECK(at::equal(direct, reference.narrow(0, prefix, count)),
                "Matched-operand rectangular suffix attention changed outputs prefix=",
                prefix,
                " maxabs=",
                (direct - reference.narrow(0, prefix, count)).abs().max().item<float>());
  }
  std::cout << "Matched-operand cached suffix attention preserves original BF16 arithmetic\n";
}

static void single_query_tiles(const c10::Device& device) {
  auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  const std::vector<int32_t> offsets{0, 65, 193, 706, 2754}, queries{0, 1, 2, 3, 4};
  auto cumulative = at::tensor(offsets, options.dtype(at::kInt));
  auto suffix_cu = at::tensor(queries, options.dtype(at::kInt));
  auto selected = at::tensor(std::vector<int64_t>{64, 192, 705, 2753}, options.dtype(at::kLong));
  for (int seed = 0; seed < 16; ++seed)
    for (int64_t kvheads : {2, 8}) {
      auto generator = at::detail::createCPUGenerator(1000 + seed);
      auto operands = at::randn({2754, 16 + 2 * kvheads, 128}, generator, at::kFloat).to(options);
      auto q = operands.narrow(1, 0, 16), k = operands.narrow(1, 16, kvheads),
           v = operands.narrow(1, 16 + kvheads, kvheads);
      auto reference =
          infinity::torch_flash_fma(q, k, v, cumulative, 2048, 1. / std::sqrt(128.), true);
      auto actual = infinity::torch_flash_fma_varlen(q.index_select(0, selected),
                                                     k,
                                                     v,
                                                     suffix_cu,
                                                     cumulative,
                                                     1,
                                                     2048,
                                                     1. / std::sqrt(128.),
                                                     true);
      TORCH_CHECK(at::equal(actual, reference.index_select(0, selected)),
                  "Single-query tile changed original BF16 arithmetic seed=",
                  seed,
                  " kvheads=",
                  kvheads);
    }
  std::cout << "Single-query BF16 ragged/GQA tiles match full-context outputs bitwise\n";
}

static void default_torch_suffix_spans(const c10::Device& device, bool extended_only = false) {
  // Query lengths exercise short spans against independent full
  // causal execution. Ragged contexts and GQA expose row/head addressing errors.
  auto opts = at::TensorOptions().device(device).dtype(at::kBFloat16);
  const std::vector<int32_t> full_offsets{0, 16, 48, 113, 306, 819, 2867};
  for (const std::vector<int32_t>& query_offsets :
       {std::vector<int32_t>{0, 2, 9, 17, 33, 57, 89},
        std::vector<int32_t>{0, 2, 9, 42, 89, 141, 205}}) {
    std::vector<int64_t> selected;
    int64_t maximum = 0;
    for (size_t row = 0; row + 1 < full_offsets.size(); ++row) {
      maximum = std::max<int64_t>(maximum, query_offsets[row + 1] - query_offsets[row]);
      for (int64_t token = full_offsets[row + 1] - (query_offsets[row + 1] - query_offsets[row]);
           token < full_offsets[row + 1];
           ++token)
        selected.push_back(token);
    }
    if (extended_only && maximum <= 32)
      continue;
    auto cu = at::tensor(full_offsets, opts.dtype(at::kInt));
    auto suffix_cu = at::tensor(query_offsets, opts.dtype(at::kInt));
    auto indices = at::tensor(selected, opts.dtype(at::kLong));
    for (int seed = 0; seed < 16; ++seed)
      for (int64_t kvheads : {2, 8}) {
        auto generator = at::detail::createCPUGenerator(2300 + seed);
        auto values =
            at::randn({full_offsets.back(), 16 + 2 * kvheads, 128}, generator, at::kFloat).to(opts);
        auto q = values.narrow(1, 0, 16), k = values.narrow(1, 16, kvheads),
             v = values.narrow(1, 16 + kvheads, kvheads);
        auto reference = infinity::torch_flash_fma(q, k, v, cu, 2048, 1. / std::sqrt(128.), true);
        auto actual = infinity::torch_flash_fma_varlen(q.index_select(0, indices),
                                                       k,
                                                       v,
                                                       suffix_cu,
                                                       cu,
                                                       maximum,
                                                       2048,
                                                       1. / std::sqrt(128.),
                                                       true);
        TORCH_CHECK(at::equal(actual, reference.index_select(0, indices)),
                    "Short-query tile changed fused BF16 attention bits seed=",
                    seed,
                    " kvheads=",
                    kvheads);
      }
  }
  if (extended_only)
    return;
  constexpr int64_t total = 2048;
  for (int64_t dim : {64, 128})
    for (auto dtype : {at::kBFloat16, at::kHalf}) {
      const int64_t heads = dim == 64 ? 14 : 16, kvheads = dim == 64 ? 2 : 8;
      auto options = at::TensorOptions().device(device).dtype(dtype);
      // Seed1244 exercises an FP16 value at an output rounding boundary.
      auto generator = at::detail::createCPUGenerator(1244);
      auto cpu = at::TensorOptions().dtype(at::kFloat);
      auto q = at::randn({total, heads, dim}, generator, cpu).to(options);
      auto k = at::randn({total, kvheads, dim}, generator, cpu).to(options);
      auto v = at::randn({total, kvheads, dim}, generator, cpu).to(options);
      auto cumulative = at::tensor(std::vector<int32_t>{0, total}, options.dtype(at::kInt));
      const auto scale = 1. / std::sqrt(double(dim));
      auto reference = std::get<0>(at::_flash_attention_forward(
          q, k, v, cumulative, cumulative, total, total, 0., true, false, scale, -1, -1));
      auto suffix_cu = at::tensor(std::vector<int32_t>{0, 1}, options.dtype(at::kInt));
      auto actual = infinity::torch_flash_unfused_varlen(
          q.narrow(0, total - 1, 1), k, v, suffix_cu, cumulative, 1, total, scale);
      auto expected = reference.narrow(0, total - 1, 1);
      bool agrees = at::equal(actual, expected);
      if (!agrees && dtype == at::kHalf && dim == 128) {
        auto a = actual.cpu().contiguous(), b = expected.cpu().contiguous();
        int maximum = 0;
        auto ordered = [](uint16_t bits) {
          return bits & 0x8000 ? 32768 - int(bits & 0x7fff) : 32768 + int(bits);
        };
        for (int64_t i = 0; i < a.numel(); ++i) {
          uint16_t x, y;
          std::memcpy(&x, a.const_data_ptr<at::Half>() + i, sizeof(x));
          std::memcpy(&y, b.const_data_ptr<at::Half>() + i, sizeof(y));
          maximum = std::max(maximum, std::abs(ordered(x) - ordered(y)));
        }
        TORCH_CHECK(
            maximum <= 1, "Default Torch head128 query exceeded one FP16 output step: ", maximum);
        agrees = true;
      }
      TORCH_CHECK(agrees,
                  "Default Torch final query changed full-context arithmetic dtype=",
                  dtype,
                  " head_dim=",
                  dim,
                  " maxabs=",
                  (actual - expected).abs().max().item<float>());
    }
  std::cout
      << "Default Torch final queries match BF16 exactly and head128 FP16 within one output step\n";
}

static void cudnn_cached_suffix(const c10::Device& device) {
  auto major = at::cuda::getDeviceProperties(device.index())->major;
  if (major != 9 && major != 10)
    return;
  constexpr int64_t tokens = 512, heads = 14, kvheads = 2, dim = 64;
  auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto q = at::randn({tokens, heads, dim}, options), k = at::randn({tokens, kvheads, dim}, options),
       v = at::randn_like(k);
  int32_t full_offsets[] = {0, tokens};
  infinity::PackedInput original;
  original.offsets = full_offsets;
  original.batch = 1;
  original.max_sequence = tokens;
  original.cumulative = at::tensor(std::vector<int32_t>{0, tokens}, options.dtype(at::kInt));
  auto reference = std::get<0>(at::_cudnn_attention_forward(q,
                                                            k,
                                                            v,
                                                            std::nullopt,
                                                            original.cumulative,
                                                            original.cumulative,
                                                            tokens,
                                                            tokens,
                                                            true,
                                                            0.,
                                                            true,
                                                            false));
  for (int64_t boundary : {1, 13, 384, 448, 511}) {
    auto count = tokens - boundary;
    auto source = std::make_shared<infinity::PrefixEntry>();
    source->layers[0] = {
        k.narrow(0, 0, boundary).clone(), v.narrow(0, 0, boundary).clone(), {}, {}};
    auto budget = std::make_shared<infinity::PrefixBudget>();
    infinity::PrefixContext context{
        original, {0, int32_t(count)}, {boundary}, {source}, {nullptr}, 16 << 20, budget};
    int32_t suffix_offsets[] = {0, int32_t(count)};
    auto input = original;
    input.offsets = suffix_offsets;
    input.cumulative = at::tensor(std::vector<int32_t>{0, int32_t(count)}, options.dtype(at::kInt));
    input.max_sequence = count;
    input.prefix = &context;
    auto actual = infinity::prefix_attention(q.narrow(0, boundary, count),
                                             k.narrow(0, boundary, count),
                                             v.narrow(0, boundary, count),
                                             input,
                                             0,
                                             1. / std::sqrt(double(dim)),
                                             -1,
                                             -1,
                                             false,
                                             true,
                                             true);
    TORCH_CHECK(actual.sizes() == reference.narrow(0, boundary, count).sizes() &&
                    at::equal(actual, reference.narrow(0, boundary, count)),
                "cuDNN cache attention changed packed-provider output boundary=",
                boundary);
  }
  std::cout << "cuDNN cached GQA suffixes preserve packed-provider outputs exactly\n";
}

// Media attention retains true head width72 and distinct query/key lengths.
static void large_local_spans(const c10::Device& device) {
  auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  for (bool grouped : {false, true})
    for (int window : {-1, 0, 2, 64}) {
      std::vector<int32_t> offsets{0};
      int maximum = 0;
      for (int i = 0; i < 32; ++i) {
        int n = 97 + (i % 5) * 17;
        offsets.push_back(offsets.back() + n);
        maximum = std::max(maximum, n);
      }
      const int heads = 4, key_heads = grouped ? 2 : 4;
      auto combined = at::randn({offsets.back(), heads + 2 * key_heads, 64}, options);
      auto q = combined.narrow(1, 0, heads), k = combined.narrow(1, heads, key_heads),
           v = combined.narrow(1, heads + key_heads, key_heads);
      auto cumulative = at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
      auto output =
          infinity::torch_flash_fma(q, k, v, cumulative, maximum, 1. / 8., false, window, window);
      for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const int start = offsets[row], n = offsets[row + 1] - start;
        auto cpu = [&](const at::Tensor& x) {
          return x.narrow(0, start, n).cpu().to(at::kDouble).transpose(0, 1);
        };
        auto qr = cpu(q), kr = cpu(k), vr = cpu(v);
        if (grouped) {
          kr = kr.repeat_interleave(2, 0);
          vr = vr.repeat_interleave(2, 0);
        }
        auto index = at::arange(n, at::kLong), delta = index.unsqueeze(0) - index.unsqueeze(1);
        auto allowed = window < 0 ? at::ones({n, n}, at::kBool) : delta.abs() <= window;
        auto probability = at::softmax(
            (at::matmul(qr, kr.transpose(1, 2)) / 8.).masked_fill(allowed.logical_not(), -INFINITY),
            -1);
        auto expected = at::matmul(probability, vr).transpose(0, 1);
        TORCH_CHECK((output.narrow(0, start, n).cpu().to(at::kDouble) - expected)
                            .abs()
                            .max()
                            .item<double>() < .04,
                    "Large local tiled attention differs from independent masked FP64 oracle");
      }
      auto changed = v.clone();
      changed.narrow(0, 0, offsets[1]).add_(20);
      auto isolated = infinity::torch_flash_fma(
          q, k, changed, cumulative, maximum, 1. / 8., false, window, window);
      TORCH_CHECK(at::equal(output.narrow(0, offsets[1], offsets.back() - offsets[1]),
                            isolated.narrow(0, offsets[1], offsets.back() - offsets[1])),
                  "Large local attention leaks across sequences");
    }
}

static void media_spans(const c10::Device& device) {
  auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto probe = at::zeros({3, 4, 72}, options);
  const auto* properties = at::cuda::getDeviceProperties(device.index());
  bool supported = properties->major >= 8 && properties->sharedMemPerBlockOptin >= 48 * 1024;
  TORCH_CHECK(infinity::torch_flash_fma_supported(probe, probe, probe) == supported,
              "Head72 architecture guard");
  TORCH_CHECK(!infinity::torch_flash_fma_supported(probe, probe, probe, true, -1, -1),
              "Head72 causal fallback");
  TORCH_CHECK(!infinity::torch_flash_fma_supported(probe, probe, probe, false, 2, 2),
              "Head72 local fallback");
  auto half = probe.to(at::kHalf), cpu = probe.cpu();
  TORCH_CHECK(!infinity::torch_flash_fma_supported(half, half, half), "Head72 FP16 fallback");
  TORCH_CHECK(!infinity::torch_flash_fma_supported(cpu, cpu, cpu), "Head72 CPU fallback");
  if (!supported)
    return;
  for (int64_t dim : {72, 256})
    for (bool causal : {false, true}) {
      if (dim == 72 && causal)
        continue;
      std::vector<int32_t> qo{0, 1, 4, 11}, ko{0, 3, 11, 24};
      auto cuq = at::tensor(qo, at::TensorOptions().dtype(at::kInt)).to(device),
           cuk = at::tensor(ko, at::TensorOptions().dtype(at::kInt)).to(device);
      auto qstore = at::randn({11, 6, dim}, options) * .1,
           kstore = at::randn({24, 4, dim}, options) * .1;
      auto q = qstore.narrow(1, 0, 4), k = kstore.narrow(1, 0, 2), v = kstore.narrow(1, 2, 2);
      auto result = infinity::torch_flash_fma_varlen(q, k, v, cuq, cuk, 7, 13, 1., causal);
      TORCH_CHECK(result.sizes() == q.sizes(),
                  "Media attention allocated synthetic rows/head width");
      double worst = 0;
      for (size_t b = 0; b < 3; ++b) {
        int64_t nq = qo[b + 1] - qo[b], nk = ko[b + 1] - ko[b];
        auto qr = q.narrow(0, qo[b], nq).cpu().to(at::kDouble).transpose(0, 1);
        auto kr =
            k.narrow(0, ko[b], nk).cpu().to(at::kDouble).transpose(0, 1).repeat_interleave(2, 0);
        auto vr =
            v.narrow(0, ko[b], nk).cpu().to(at::kDouble).transpose(0, 1).repeat_interleave(2, 0);
        auto scores = at::matmul(qr, kr.transpose(1, 2));
        if (causal) {
          auto delta = at::arange(nk, at::kLong).unsqueeze(0) -
                       at::arange(nq, at::kLong).unsqueeze(1) - (nk - nq);
          scores = scores.masked_fill(delta > 0, -INFINITY);
        }
        auto expected = at::matmul(at::softmax(scores, -1), vr).transpose(0, 1);
        worst = std::max(worst,
                         (result.narrow(0, qo[b], nq).cpu().to(at::kDouble) - expected)
                             .abs()
                             .max()
                             .item<double>());
      }
      TORCH_CHECK(worst < .003, "Unequal media spans FP64 reference", worst);
      auto changed = v.clone();
      changed.narrow(0, 0, 3).add_(1);
      auto isolated = infinity::torch_flash_fma_varlen(q, k, changed, cuq, cuk, 7, 13, 1., causal);
      TORCH_CHECK(at::equal(result.narrow(0, 1, 10), isolated.narrow(0, 1, 10)),
                  "Unequal media span sequence isolation");
    }
}

// Independent reference computes the bias from real per-sequence positions;
// it does not share the Flash kernel's pre-scaled ALiBi implementation.
static void alibi32_spans(const c10::Device& device) {
  const bool supported = at::cuda::getDeviceProperties(device.index())->major >= 8;
  TORCH_CHECK(infinity::torch_flash_fma_alibi32_supported(device, at::kHalf, 32) == supported,
              "Head32 ALiBi architecture guard mismatch");
  TORCH_CHECK(!infinity::torch_flash_fma_alibi32_supported(c10::Device(at::kCPU), at::kHalf, 32) &&
                  !infinity::torch_flash_fma_alibi32_supported(device, at::kBFloat16, 32) &&
                  !infinity::torch_flash_fma_alibi32_supported(device, at::kFloat, 32) &&
                  !infinity::torch_flash_fma_alibi32_supported(device, at::kHalf, 64),
              "Head32 ALiBi unsupported geometry/dtype must retain fallback");
  if (!supported)
    return;
  const int64_t heads = 12, dim = 32;
  // Original non-power-of-two Jina slope order, with a zero-slope case too.
  std::vector<float> values{.5f,
                            .25f,
                            .125f,
                            .0625f,
                            .03125f,
                            .015625f,
                            .0078125f,
                            .00390625f,
                            static_cast<float>(std::sqrt(.5)),
                            static_cast<float>(std::pow(.5, 1.5)),
                            static_cast<float>(std::pow(.5, 2.5)),
                            static_cast<float>(std::pow(.5, 3.5))};
  auto options = at::TensorOptions().device(device).dtype(at::kHalf);
  for (bool zero_slope : {false, true})
    for (const auto& lengths : {std::vector<int32_t>{3, 1, 17}, std::vector<int32_t>{32, 129, 7}}) {
      auto slopes = at::tensor(values, at::TensorOptions().dtype(at::kFloat)).to(device);
      if (zero_slope)
        slopes.zero_();
      std::vector<int32_t> offsets{0};
      int64_t total = 0, maximum = 0;
      for (auto length : lengths) {
        total += length;
        maximum = std::max<int64_t>(maximum, length);
        offsets.push_back(total);
      }
      auto combined = at::randn({total, 3 * heads, dim}, options);
      auto q = combined.narrow(1, 0, heads), k = combined.narrow(1, heads, heads),
           v = combined.narrow(1, 2 * heads, heads);
      auto cumulative = at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
      const double scale = 1. / std::sqrt(static_cast<double>(dim));
      auto result = infinity::torch_flash_fma_alibi32(q, k, v, cumulative, slopes, maximum, scale);
      TORCH_CHECK(result.sizes() == q.sizes() && result.numel() == total * heads * dim &&
                      at::isfinite(result).all().item<bool>(),
                  "Head32 ALiBi changed real token/head geometry or finiteness");
      double worst = 0;
      auto cpu_slopes = slopes.cpu().to(at::kDouble).view({heads, 1, 1});
      for (size_t b = 0; b < lengths.size(); ++b) {
        auto cpu = [&](const at::Tensor& t) {
          return t.narrow(0, offsets[b], lengths[b]).cpu().to(at::kDouble).transpose(0, 1);
        };
        auto qr = cpu(q), kr = cpu(k), vr = cpu(v), positions = at::arange(lengths[b], at::kDouble);
        auto distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs();
        auto scores = at::matmul(qr, kr.transpose(1, 2)) * scale - cpu_slopes * distance;
        auto expected = at::matmul(at::softmax(scores, -1), vr).transpose(0, 1);
        worst = std::max(worst,
                         (result.narrow(0, offsets[b], lengths[b]).cpu().to(at::kDouble) - expected)
                             .abs()
                             .max()
                             .item<double>());
      }
      TORCH_CHECK(worst < .004, "Head32 ALiBi independent FP64 bias/attention mismatch: ", worst);
      auto changed = v.clone();
      changed.select(0, lengths[0] - 1).add_(5.);
      auto isolated =
          infinity::torch_flash_fma_alibi32(q, k, changed, cumulative, slopes, maximum, scale);
      TORCH_CHECK(at::equal(result.narrow(0, lengths[0], total - lengths[0]),
                            isolated.narrow(0, lengths[0], total - lengths[0])),
                  "Head32 ALiBi packed sequence isolation failed");
      TORCH_CHECK(!at::equal(result[0], isolated[0]),
                  "Head32 ALiBi unexpectedly masked future tokens");
      std::cout << "FP16 head32 ALiBi total=" << total << " zero_slope=" << zero_slope
                << " maxabs=" << worst << '\n';
    }
}

int main(int argc, char** argv) {
  c10::InferenceMode guard;
  if (argc > 1 && std::string(argv[1]) == "suffix64") {
    default_torch_suffix_spans(c10::Device("cuda:0"), true);
    return 0;
  }
  const c10::Device device(argc > 1 ? argv[1] : "cuda:0");
  media_spans(device);
  large_local_spans(device);
  cached_suffix_spans(device);
  single_query_tiles(device);
  default_torch_suffix_spans(device);
  cudnn_cached_suffix(device);
  at::manual_seed(9917);
  at::set_num_threads(1);
  alibi32_spans(device);
  for (auto dtype : {at::kHalf, at::kBFloat16})
    for (int64_t dim : {32, 64, 128, 256})
      for (bool grouped : {false, true})
        for (bool causal : {false, true})
          for (int64_t window : {-1, 2})
            for (auto lengths :
                 {std::vector<int32_t>{3, 13, 7}, std::vector<int32_t>{32, 129, 17}}) {
              // Dedicated head32 profile keeps the generic adapter's guards unchanged.
              if (dim == 32 && (grouped || causal || window >= 0 ||
                                at::cuda::getDeviceProperties(device.index())->major != 9))
                continue;
              const int64_t heads = dim == 32 ? 12 : 4, key_heads = grouped ? 2 : heads;
              std::vector<int32_t> offsets{0};
              int64_t total = 0, max_sequence = 0;
              for (auto n : lengths) {
                total += n;
                max_sequence = std::max<int64_t>(max_sequence, n);
                offsets.push_back(total);
              }
              auto options = at::TensorOptions().device(device).dtype(dtype);
              // Fresh fused QKV storage tests noncontiguous token/head views, including GQA.
              auto combined = at::randn({total, heads + 2 * key_heads, dim}, options);
              auto q = combined.narrow(1, 0, heads), k = combined.narrow(1, heads, key_heads),
                   v = combined.narrow(1, heads + key_heads, key_heads);
              auto cu = at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
              const double scale = 1. / std::sqrt(dim);
              auto output = dim == 32
                                ? infinity::torch_flash_exact32(q, k, v, cu, max_sequence, scale)
                                : infinity::torch_flash_fma(
                                      q, k, v, cu, max_sequence, scale, causal, window, window);
              TORCH_CHECK(at::isfinite(output).all().item<bool>(),
                          "Nonfinite native Torch Flash output");
              double worst = 0;
              for (size_t row = 0; row + 1 < offsets.size(); ++row) {
                const int64_t start = offsets[row], n = offsets[row + 1] - start;
                auto cpu = [&](const at::Tensor& x) {
                  return x.narrow(0, start, n)
                      .to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble))
                      .transpose(0, 1);
                };
                auto qr = cpu(q), kr = cpu(k), vr = cpu(v);
                if (grouped) {
                  kr = kr.repeat_interleave(heads / key_heads, 0);
                  vr = vr.repeat_interleave(heads / key_heads, 0);
                }
                auto index = at::arange(n, at::kLong),
                     delta = index.unsqueeze(0) - index.unsqueeze(1);
                auto allowed = at::ones({n, n}, at::kBool);
                if (causal)
                  allowed.logical_and_(delta <= 0);
                if (window >= 0)
                  allowed.logical_and_(delta >= -window).logical_and_(delta <= window);
                auto probabilities = at::softmax((at::matmul(qr, kr.transpose(1, 2)) * scale)
                                                     .masked_fill(allowed.logical_not(), -INFINITY),
                                                 -1);
                auto expected = at::matmul(probabilities, vr).transpose(0, 1);
                worst = std::max(worst,
                                 (output.narrow(0, start, n)
                                      .to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble)) -
                                  expected)
                                     .abs()
                                     .max()
                                     .item<double>());
              }
              TORCH_CHECK(worst < (dtype == at::kHalf ? .004 : .04),
                          "Native Torch Flash reference mismatch: ",
                          worst);
              auto changed = v.clone();
              changed.narrow(0, 0, lengths[0]).add_(20);
              auto isolated =
                  dim == 32 ? infinity::torch_flash_exact32(q, k, changed, cu, max_sequence, scale)
                            : infinity::torch_flash_fma(
                                  q, k, changed, cu, max_sequence, scale, causal, window, window);
              TORCH_CHECK(at::isfinite(isolated).all().item<bool>() &&
                              at::equal(output.narrow(0, lengths[0], total - lengths[0]),
                                        isolated.narrow(0, lengths[0], total - lengths[0])),
                          "Packed sequence isolation failed");
              std::cout << "dtype=" << dtype << " total=" << total << " dim=" << dim
                        << " grouped=" << grouped << " causal=" << causal << " window=" << window
                        << " maxabs=" << worst << '\n';
            }
  // The exact-width512 launch is intentionally restricted; callers retain
  // efficient attention for unsupported devices, masks and activation types.
  auto wide_options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto probe = at::zeros({3, 1, 512}, wide_options);
  const auto* properties = at::cuda::getDeviceProperties(device.index());
  const bool hopper = properties->major >= 8 && properties->sharedMemPerBlockOptin >= 128 * 1024;
  TORCH_CHECK(infinity::torch_flash_fma_supported(probe, probe, probe, false, -1, -1) == hopper,
              "Wide-head architecture guard mismatch");
  TORCH_CHECK(infinity::torch_flash_fma_supported(probe, probe, probe, true, -1, -1) == hopper,
              "Wide-head causal architecture guard mismatch");
  TORCH_CHECK(!infinity::torch_flash_fma_supported(probe, probe, probe, false, 2, 2),
              "Wide-head local mask must retain fallback");
  auto half_probe = probe.to(at::kHalf), cpu_probe = probe.cpu();
  TORCH_CHECK(
      !infinity::torch_flash_fma_supported(half_probe, half_probe, half_probe, false, -1, -1),
      "Wide-head FP16 must retain fallback");
  TORCH_CHECK(!infinity::torch_flash_fma_supported(cpu_probe, cpu_probe, cpu_probe, false, -1, -1),
              "Wide-head CPU must retain fallback");
  if (hopper)
    for (bool causal : {false, true})
      for (bool grouped : {false, true})
        for (auto lengths : {std::vector<int32_t>{3, 17, 9}, std::vector<int32_t>{32, 129, 17}}) {
          const int64_t heads = 4, key_heads = grouped ? 1 : 4;
          std::vector<int32_t> offsets{0};
          int64_t total = 0, maximum = 0;
          for (auto n : lengths) {
            total += n;
            maximum = std::max<int64_t>(maximum, n);
            offsets.push_back(total);
          }
          auto combined = at::randn({total, heads + 2 * key_heads, 512}, wide_options) * .1;
          auto q = combined.narrow(1, 0, heads), k = combined.narrow(1, heads, key_heads),
               v = combined.narrow(1, heads + key_heads, key_heads);
          auto cu = at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
          auto output = infinity::torch_flash_fma(q, k, v, cu, maximum, 1., causal, -1, -1);
          TORCH_CHECK(output.sizes() == q.sizes() && at::isfinite(output).all().item<bool>(),
                      "Wide-head packed result geometry/finiteness mismatch");
          double worst = 0;
          for (size_t row = 0; row + 1 < offsets.size(); ++row) {
            int64_t start = offsets[row], n = offsets[row + 1] - start;
            auto cpu = [&](const at::Tensor& x) {
              return x.narrow(0, start, n)
                  .to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble))
                  .transpose(0, 1);
            };
            auto qr = cpu(q), kr = cpu(k), vr = cpu(v);
            if (grouped) {
              kr = kr.repeat_interleave(heads / key_heads, 0);
              vr = vr.repeat_interleave(heads / key_heads, 0);
            }
            auto scores = at::matmul(qr, kr.transpose(1, 2));
            if (causal)
              scores = scores.masked_fill(
                  at::ones({n, n}, at::TensorOptions().dtype(at::kBool)).triu(1), -INFINITY);
            auto expected = at::matmul(at::softmax(scores, -1), vr).transpose(0, 1);
            worst = std::max(worst,
                             (output.narrow(0, start, n).cpu().to(at::kDouble) - expected)
                                 .abs()
                                 .max()
                                 .item<double>());
          }
          TORCH_CHECK(worst < .003, "Wide-head independent FP64 ragged/GQA mismatch: ", worst);
          auto changed = v.clone();
          changed.narrow(0, 0, lengths[0]).add_(1.);
          auto isolated = infinity::torch_flash_fma(q, k, changed, cu, maximum, 1., causal, -1, -1);
          TORCH_CHECK(at::equal(output.narrow(0, lengths[0], total - lengths[0]),
                                isolated.narrow(0, lengths[0], total - lengths[0])),
                      "Wide-head packed sequence isolation failed");
          if (causal) {
            auto one = at::tensor(std::vector<int32_t>{0, 1}, at::TensorOptions().dtype(at::kInt))
                           .to(device);
            auto prefix = infinity::torch_flash_fma(
                q.narrow(0, 0, 1), k.narrow(0, 0, 1), v.narrow(0, 0, 1), one, 1, 1., true, -1, -1);
            TORCH_CHECK(at::equal(prefix, output.narrow(0, 0, 1)),
                        "Wide-head causal prefix preservation failed");
          }
          std::cout << "BF16 head512 ragged/GQA total=" << total << " grouped=" << grouped
                    << " causal=" << causal << " maxabs=" << worst << '\n';
        }
}
