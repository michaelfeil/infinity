#include "kernels/qwen35_reference.h"
#include "kernels/flashinfer_gdn.h"
#include "helpers/fp8_linear.h"
#include "kernels/qwen35_recurrent.h"
#include <array>
#include "kernels/qwen35_kernels.h"
#include "batching/prefix_cache.h"
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

void resumed_delta(c10::Device device,
                   int64_t kh,
                   int64_t vh,
                   bool fma,
                   bool quantization_only = false,
                   int32_t tokens = 35) {
  TORCH_CHECK(tokens > 13, "Split/resume fixture requires at least 14 tokens");
  auto opts = at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto qkv = (at::randn({tokens, (2 * kh + vh) * 128}, opts) * .2).contiguous();
  auto z = (at::randn({tokens, vh * 128}, opts) * .2).contiguous();
  auto ab = (at::randn({tokens, 2 * vh}, opts) * .2).contiguous();
  auto conv = (at::randn({(2 * kh + vh) * 128, 4}, opts) * .2).contiguous();
  auto alog = at::zeros({vh}, opts.dtype(at::kFloat)), dt = at::zeros_like(alog);
  auto norm =
      quantization_only ? (at::randn({128}, opts) * .2 + 1).contiguous() : at::ones({128}, opts);
  if (quantization_only)
    qkv[0].zero_();
  int32_t offsets[] = {0, tokens};
  infinity::PackedInput full;
  full.offsets = offsets;
  full.batch = 1;
  full.max_sequence = tokens;
  full.cumulative = at::tensor(std::vector<int32_t>{0, tokens}, opts.dtype(at::kInt));
  auto run = [&](const infinity::PackedInput& input,
                 int64_t start,
                 const std::pair<at::Tensor, at::Tensor>* quantized = nullptr) {
    auto count = tokens - start;
    if (fma)
      return infinity::qwen35_delta_cuda(qkv.narrow(0, start, count),
                                         z.narrow(0, start, count),
                                         ab.narrow(0, start, count),
                                         conv,
                                         alog,
                                         dt,
                                         norm,
                                         input,
                                         kh,
                                         vh,
                                         1e-6,
                                         {},
                                         nullptr,
                                         {},
                                         {},
                                         false,
                                         0,
                                         false,
                                         quantized);
    return infinity::qwen35_reference_delta_cuda(qkv.narrow(0, start, count),
                                                 z.narrow(0, start, count),
                                                 ab.narrow(0, start, count),
                                                 conv,
                                                 alog,
                                                 dt,
                                                 norm,
                                                 input,
                                                 kh,
                                                 vh,
                                                 1e-6,
                                                 0);
  };
  auto expected = run(full, 0);
  if (fma && kh == 16 && vh == 16) {
    // Merged QKV/Z projections share storage. Poison the surrounding columns so
    // contiguous indexing or a lost slice offset changes the outputs.
    const auto qkv_width = qkv.size(1), gate_width = z.size(1);
    auto packed = at::full({tokens, qkv_width + gate_width + 8}, NAN, opts);
    auto queries = packed.narrow(1, 4, qkv_width);
    auto gates = packed.narrow(1, 4 + qkv_width, gate_width);
    queries.copy_(qkv);
    gates.copy_(z);
    qkv = queries;
    z = gates;
    TORCH_CHECK(at::equal(run(full, 0), expected),
                "Strided QKV/gates changed packed DeltaNet output");
  }
  if (fma) {
    auto values = at::ones({tokens + 2, vh * 128}, opts).to(at::kFloat8_e4m3fn);
    auto scales = at::full({tokens + 2, 1}, -7.f, opts.dtype(at::kFloat));
    auto sentinel = values[0].view(at::kByte).clone();
    std::pair<at::Tensor, at::Tensor> quantized = {values.narrow(0, 1, tokens),
                                                   scales.narrow(0, 1, tokens)};
    run(full, 0, &quantized);
    auto separate = infinity::fp8_quantize_rows(expected);
    TORCH_CHECK(at::equal(quantized.first.view(at::kByte), separate.first.view(at::kByte)) &&
                    at::equal(quantized.second, separate.second) &&
                    at::equal(values[0].view(at::kByte), sentinel) &&
                    at::equal(values[tokens + 1].view(at::kByte), sentinel) &&
                    scales[0].item<float>() == -7.f && scales[tokens + 1].item<float>() == -7.f,
                "Fused GDN normalization changed FP8 bytes/scales or wrote outside its slice");
    if (vh == 1) {
      auto storage = at::empty({tokens * vh * 128 + 1}, values.options());
      std::pair<at::Tensor, at::Tensor> odd = {
          storage.narrow(0, 1, tokens * vh * 128).view({tokens, vh * 128}), quantized.second};
      bool rejected = false;
      try {
        run(full, 0, &odd);
      } catch (const c10::Error&) {
        rejected = true;
      }
      TORCH_CHECK(rejected, "Misaligned FP8 output must fail before launching a CUDA kernel");
    }
  }
  if (quantization_only)
    return;
  for (int64_t boundary : {1, 13, tokens - 1}) {
    if (fma) {
      auto prefill = full;
      int32_t prefix_offsets[] = {0, int32_t(boundary)};
      prefill.offsets = prefix_offsets;
      prefill.max_sequence = boundary;
      prefill.cumulative =
          at::tensor(std::vector<int32_t>{0, int32_t(boundary)}, opts.dtype(at::kInt));
      at::Tensor state;
      auto prefix = infinity::qwen35_delta_cuda(qkv.narrow(0, 0, boundary),
                                                z.narrow(0, 0, boundary),
                                                ab.narrow(0, 0, boundary),
                                                conv,
                                                alog,
                                                dt,
                                                norm,
                                                prefill,
                                                kh,
                                                vh,
                                                1e-6,
                                                {},
                                                &state,
                                                {},
                                                {},
                                                false);
      // Tensor-core prefill rounds Q/K to BF16, while scalar recurrence retains
      // FP32 normalized Q/K. Bound that provider drift; state sharing below must
      // still preserve its inputs and repeat exactly across batch rows.
      TORCH_CHECK(at::allclose(prefix, expected.narrow(0, 0, boundary), 1e-2, 2e-3),
                  "Block prefill exceeded BF16 recurrent tolerance");
      if (boundary == 13) {
        auto storage = at::empty({state.numel() + 1}, state.options());
        auto shifted = storage.narrow(0, 1, state.numel()).view(state.sizes());
        shifted.copy_(state);
        state = shifted;
      }
      auto immutable = state.clone();
      auto suffix = full;
      int32_t suffix_offsets[] = {0, int32_t(tokens - boundary)};
      suffix.offsets = suffix_offsets;
      suffix.max_sequence = tokens - boundary;
      suffix.cumulative =
          at::tensor(std::vector<int32_t>{0, int32_t(tokens - boundary)}, opts.dtype(at::kInt));
      suffix.block_prefix_sources = {0};
      suffix.block_cache_sources = at::zeros({1}, opts.dtype(at::kLong));
      auto history =
          qkv.narrow(0, std::max(int64_t(0), boundary - 3), std::min(boundary, int64_t(3)))
              .contiguous();
      auto history_cu =
          at::tensor(std::vector<int32_t>{0, int32_t(history.size(0))}, opts.dtype(at::kInt));
      at::Tensor final_state;
      auto resumed = infinity::qwen35_delta_cuda(qkv.narrow(0, boundary, tokens - boundary),
                                                 z.narrow(0, boundary, tokens - boundary),
                                                 ab.narrow(0, boundary, tokens - boundary),
                                                 conv,
                                                 alog,
                                                 dt,
                                                 norm,
                                                 suffix,
                                                 kh,
                                                 vh,
                                                 1e-6,
                                                 state,
                                                 &final_state,
                                                 history,
                                                 history_cu,
                                                 false);
      TORCH_CHECK(
          at::allclose(resumed, expected.narrow(0, boundary, tokens - boundary), 1e-2, 2e-3) &&
              at::equal(state, immutable),
          "Block resume changed outputs or mutated shared state");
      // Provider capture exposes native-logical K/V over provider V/K storage. Resuming from
      // that stride layout must match the native-contiguous state bit-for-bit.
      auto provider_state = state.transpose(2, 3).contiguous().transpose(2, 3);
      auto provider_immutable = provider_state.clone();
      at::Tensor provider_final;
      auto provider_resumed =
          infinity::qwen35_delta_cuda(qkv.narrow(0, boundary, tokens - boundary),
                                      z.narrow(0, boundary, tokens - boundary),
                                      ab.narrow(0, boundary, tokens - boundary),
                                      conv,
                                      alog,
                                      dt,
                                      norm,
                                      suffix,
                                      kh,
                                      vh,
                                      1e-6,
                                      provider_state,
                                      &provider_final,
                                      history,
                                      history_cu,
                                      false);
      TORCH_CHECK(at::equal(provider_resumed, resumed) && at::equal(provider_final, final_state) &&
                      at::equal(provider_state, provider_immutable),
                  "Provider-layout state view changed resumed output or mutated prefix state");
      auto batched = suffix;
      std::vector<int32_t> batch_offsets;
      for (int i = 0; i <= 8; ++i)
        batch_offsets.push_back(i * (tokens - boundary));
      batched.batch = 8;
      batched.offsets = batch_offsets.data();
      batched.cumulative = at::tensor(batch_offsets, opts.dtype(at::kInt));
      batched.block_prefix_sources.assign(8, 0);
      batched.block_cache_sources = at::zeros({8}, opts.dtype(at::kLong));
      at::Tensor batch_state;
      auto repeated =
          infinity::qwen35_delta_cuda(qkv.narrow(0, boundary, tokens - boundary).repeat({8, 1}),
                                      z.narrow(0, boundary, tokens - boundary).repeat({8, 1}),
                                      ab.narrow(0, boundary, tokens - boundary).repeat({8, 1}),
                                      conv,
                                      alog,
                                      dt,
                                      norm,
                                      batched,
                                      kh,
                                      vh,
                                      1e-6,
                                      state,
                                      &batch_state,
                                      history,
                                      history_cu,
                                      false);
      TORCH_CHECK(at::equal(repeated, resumed.repeat({8, 1})) &&
                      at::equal(batch_state, final_state.repeat({8, 1, 1, 1})) &&
                      at::equal(state, immutable),
                  "Batched recurrence changed outputs, final states or shared prefix");
    }
    auto budget = std::make_shared<infinity::PrefixBudget>();
    auto saved = std::make_shared<infinity::PrefixEntry>();
    saved->budget = budget;
    // This kernel-level fixture supplies the admission reservation normally
    // established by PrefixCache before requesting any checkpoint buffers.
    saved->reserved_bytes = budget->reserved_bytes = 64 << 20;
    saved->identity.tokens.resize(boundary);
    infinity::PrefixContext capture{full, {0, tokens}, {0}, {nullptr}, {saved}, 64 << 20, budget};
    auto cold = full;
    cold.prefix = &capture;
    TORCH_CHECK(at::equal(run(cold, 0), expected),
                "Cache capture changed DeltaNet arithmetic kh=",
                kh,
                " vh=",
                vh,
                " fma=",
                fma);
    auto state = saved->layers.find(0);
    TORCH_CHECK(state != saved->layers.end() && state->second.recurrent.defined(),
                "DeltaNet checkpoint was not exported");
    int32_t suffix_offsets[] = {0, int32_t(tokens - boundary)};
    infinity::PrefixContext resume{
        full, {0, int32_t(tokens - boundary)}, {boundary}, {saved}, {nullptr}, 64 << 20, budget};
    auto suffix = full;
    suffix.offsets = suffix_offsets;
    suffix.max_sequence = tokens - boundary;
    suffix.prefix = &resume;
    suffix.cumulative =
        at::tensor(std::vector<int32_t>{0, int32_t(tokens - boundary)}, opts.dtype(at::kInt));
    auto actual = run(suffix, boundary);
    TORCH_CHECK(
        at::equal(actual, expected.narrow(0, boundary, tokens - boundary)),
        "Resumed DeltaNet changed outputs kh=",
        kh,
        " vh=",
        vh,
        " fma=",
        fma,
        " boundary=",
        boundary,
        " maxabs=",
        (actual - expected.narrow(0, boundary, tokens - boundary)).abs().max().item<float>());
  }
}

// CPU oracle for the original 32-lane recurrence; does not call a CUDA dispatcher.
void independent_recurrent(c10::Device device, int kh, int vh, bool fma) {
  if (!infinity::qwen35_recurrent_supported(device, kh, vh))
    return;
  constexpr int tokens = 13;
  auto values = (at::randn({tokens, (2 * kh + vh) * 128}) * .2).to(at::kBFloat16);
  auto qk = at::randn({tokens * kh, 256}) * .05, gates = at::rand({tokens, 2 * vh});
  auto expected = at::empty({tokens, vh, 128}, values.options());
  auto v = values.const_data_ptr<at::BFloat16>();
  auto q = qk.const_data_ptr<float>(), g = gates.const_data_ptr<float>();
  auto out = expected.mutable_data_ptr<at::BFloat16>();
  auto accumulate = [&](float sum, float a, float b) {
    if (fma)
      return std::fma(a, b, sum);
    volatile float product = a * b;
    return sum + product;
  };
  auto reduce = [](std::array<float, 32> partial) {
    for (int d = 16; d; d /= 2)
      for (int lane = 0; lane < d; ++lane)
        partial[lane] += partial[lane + d];
    return partial[0];
  };
  for (int head = 0; head < vh; ++head)
    for (int column = 0; column < 128; ++column) {
      std::array<float, 128> state{};
      for (int t = 0; t < tokens; ++t) {
        const auto* query = q + (t * kh + head / (vh / kh)) * 256;
        std::array<float, 32> partial{};
        for (int lane = 0; lane < 32; ++lane)
          for (int i = 0; i < 4; ++i) {
            int k = lane + i * 32;
            state[k] *= g[t * 2 * vh + head];
            partial[lane] = accumulate(partial[lane], state[k], query[128 + k]);
          }
        volatile float difference =
            float(v[t * (2 * kh + vh) * 128 + 2 * kh * 128 + head * 128 + column]) -
            reduce(partial);
        float delta = difference * g[t * 2 * vh + vh + head];
        partial.fill(0.f);
        for (int lane = 0; lane < 32; ++lane)
          for (int i = 0; i < 4; ++i) {
            int k = lane + i * 32;
            state[k] = accumulate(state[k], query[128 + k], delta);
            partial[lane] = accumulate(partial[lane], state[k], query[k]);
          }
        out[(t * vh + head) * 128 + column] = at::BFloat16(reduce(partial));
      }
    }
  int32_t offsets[] = {0, tokens};
  infinity::PackedInput input;
  input.offsets = offsets;
  input.batch = 1;
  input.max_sequence = tokens;
  input.cumulative = at::tensor(std::vector<int32_t>{0, tokens},
                                at::TensorOptions().dtype(at::kInt).device(device));
  auto actual = at::empty_like(expected, expected.options().device(device));
  infinity::qwen35_recurrent_cuda(
      values.to(device), qk.to(device), gates.to(device), input, actual, kh, vh, fma);
  TORCH_CHECK(at::equal(actual.cpu(), expected),
              "Independent recurrence oracle failed kh=",
              kh,
              " vh=",
              vh,
              " fma=",
              fma);
}

at::Tensor independent_norm(const at::Tensor& x, const at::Tensor& gamma, float eps) {
  auto f = x.to(at::kFloat).contiguous(), g = gamma.to(at::kFloat).contiguous();
  auto result = at::empty_like(f);
  auto width = x.size(-1), rows = x.numel() / width;
  auto input = f.const_data_ptr<float>(), weight = g.const_data_ptr<float>();
  auto out = result.mutable_data_ptr<float>();
  int threads = 1;
  while (threads < width && threads < 1024)
    threads *= 2;
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<float> sums(threads, 0.f);
    for (int lane = 0; lane < threads; ++lane)
      for (int64_t col = lane; col < width; col += threads) {
        volatile float square = input[row * width + col] * input[row * width + col];
        volatile float add = sums[lane] + square;
        sums[lane] = add;
      }
    for (int stride = threads / 2; stride; stride /= 2)
      for (int lane = 0; lane < stride; ++lane) {
        volatile float add = sums[lane] + sums[lane + stride];
        sums[lane] = add;
      }
    volatile float mean = sums[0] * static_cast<float>(1. / static_cast<double>(width));
    volatile float variance = mean + eps;
    volatile float inv = 1.f / std::sqrt(variance);
    for (int64_t col = 0; col < width; ++col) {
      volatile float normalized = input[row * width + col] * inv;
      volatile float scale = weight[col] + 1.f;
      volatile float value = normalized * scale;
      out[row * width + col] = value;
    }
  }
  return result.to(at::kBFloat16);
}

int main(int argc, char** argv) {
  c10::Device device(argc > 1 ? argv[1] : "cuda:0");
  at::manual_seed(717);
  for (auto dtype : {at::kHalf, at::kBFloat16})
    for (auto shape : {std::array<int64_t, 3>{1, 1, 1}, {17, 2, 4}, {255, 16, 16}}) {
      const auto [tokens, kh, vh] = shape;
      const auto channels = (2 * kh + vh) * 128;
      auto options = at::TensorOptions().device(device);
      auto values = at::randn({tokens, channels + 8}, options.dtype(dtype)).narrow(1, 0, channels);
      auto qk = at::randn({tokens * kh, 264}, options).narrow(1, 0, 256);
      auto gates = at::randn({tokens, 2 * vh + 1}, options).narrow(1, 0, 2 * vh);
      auto packed = infinity::flashinfer_gdn_pack_cuda(values, qk, gates, kh, vh);
      auto pairs = qk.view({tokens, kh, 256});
      const std::array<at::Tensor, 5> expected = {
          pairs.narrow(2, 0, 128).to(dtype).contiguous(),
          pairs.narrow(2, 128, 128).to(dtype).contiguous(),
          values.narrow(1, 2 * kh * 128, vh * 128).contiguous(),
          gates.narrow(1, 0, vh).contiguous(),
          gates.narrow(1, vh, vh).contiguous()};
      for (size_t i = 0; i < packed.size(); ++i)
        TORCH_CHECK(at::equal(packed[i], expected[i]) && packed[i].is_contiguous() &&
                        reinterpret_cast<uintptr_t>(packed[i].const_data_ptr()) % 16 == 0,
                    "GDN packing changed provider values, casts or alignment");
    }
  std::cout << "GDN provider packing values, casts, strides and alignment passed\n";
  // Private CPU generator: this fixture must not perturb or depend on the
  // default CPU/CUDA random streams consumed by later reference cases.
  auto generator = at::detail::createCPUGenerator(0x47444e);
  auto cpu_options = at::TensorOptions().dtype(at::kBFloat16);
  auto fill_gates = [](at::Tensor ab, int64_t tokens, int64_t vh) {
    for (int64_t t = 0; t < tokens; ++t) {
      float a = 0.f, b = 0.f;
      switch (t % 5) {
      case 0:
        a = 25.f;
        b = 40.f;
        break;
      case 1:
        a = 20.f;
        b = -40.f;
        break;
      case 2:
        a = -25.f;
        b = 0.f;
        break;
      case 3:
        a = 0.f;
        b = 20.f;
        break;
      default:
        a = 1e-3f;
        b = -20.f;
        break;
      }
      ab.select(0, t).narrow(0, 0, vh).fill_(a);
      ab.select(0, t).narrow(0, vh, vh).fill_(b);
    }
  };
  // The fused fresh-prefix producer must match convolution, prepare, and pack
  // bit for bit, including padded strides, sequence boundaries, and both
  // softplus branches.
  for (auto [kh, vh] : {std::pair<int64_t, int64_t>{3, 3}, {16, 48}})
    for (int64_t tokens : {1, 257}) {
      auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
      const auto channels = (2 * kh + vh) * 128;
      auto values_base = at::randn({tokens, channels + 8}, generator, cpu_options).to(device);
      auto values = values_base.narrow(1, 4, channels);
      auto ab_base = at::randn({tokens, 2 * vh + 3}, generator, cpu_options).to(device);
      auto ab = ab_base.narrow(1, tokens % 2, 2 * vh);
      fill_gates(ab, tokens, vh);
      auto weights =
          (at::randn({channels, 4}, generator, cpu_options) * .2).contiguous().to(device);
      std::vector<int32_t> boundaries =
          tokens > 4 ? std::vector<int32_t>{0, 1, int32_t(tokens / 2), int32_t(tokens)}
                     : std::vector<int32_t>{0, int32_t(tokens)};
      auto cumulative = at::tensor(boundaries, options.dtype(at::kInt));
      auto convolution = infinity::qwen35_conv_cuda(
          values.clone(), weights, cumulative, int64_t(boundaries.size()) - 1);
      auto a_log = at::linspace(-2.f, 2.f, vh, options.dtype(at::kFloat));
      auto dt_bias = at::linspace(-.5f, .5f, vh, options.dtype(at::kFloat));
      auto fused = infinity::qwen35_gdn_prepare_pack_cuda(
          values, weights, cumulative, ab, a_log, dt_bias, kh, vh);
      auto [qk, decay] =
          infinity::qwen35_gdn_prepare_cuda(convolution, ab.contiguous(), a_log, dt_bias, kh, vh);
      auto qk_base = at::empty({tokens * kh, 264}, options.dtype(at::kFloat));
      auto qk_strided = qk_base.narrow(1, 0, 256);
      qk_strided.copy_(qk);
      auto gate_base = at::empty({tokens, 2 * vh + 3}, options.dtype(at::kFloat));
      auto gate_strided = gate_base.narrow(1, 0, 2 * vh);
      gate_strided.copy_(decay);
      auto expected =
          infinity::flashinfer_gdn_pack_cuda(convolution, qk_strided, gate_strided, kh, vh);
      for (size_t i = 0; i < fused.size(); ++i)
        TORCH_CHECK(fused[i].is_contiguous() &&
                        reinterpret_cast<uintptr_t>(fused[i].const_data_ptr()) % 16 == 0 &&
                        at::equal(fused[i].view(at::kByte), expected[i].view(at::kByte)),
                    "Fused GDN convolution/prepare/pack changed provider buffers kh=",
                    kh,
                    " vh=",
                    vh,
                    " tokens=",
                    tokens,
                    " buffer=",
                    i);
    }
  // Cached suffixes must produce the same provider buffers from history/source
  // mapping. The cases cover 1/57/185 suffix tokens, empty/partial history,
  // multiple prefixes, repeated sources, padded QKV rows, and GQA 4/6.
  auto check_cached = [&](int64_t kh,
                          int64_t vh,
                          const std::vector<int64_t>& lengths,
                          const std::vector<int64_t>& history_lengths,
                          const std::vector<int64_t>& source_ids) {
    TORCH_CHECK(lengths.size() == source_ids.size() && !history_lengths.empty(),
                "Invalid cached producer fixture");
    const auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
    const auto channels = (2 * kh + vh) * 128;
    int64_t tokens = 0, max_sequence = 0;
    std::vector<int32_t> cumulative{0};
    for (const auto length : lengths) {
      tokens += length;
      max_sequence = std::max(max_sequence, length);
      cumulative.push_back(cumulative.back() + int32_t(length));
    }
    std::vector<int32_t> prefix_cumulative{0};
    for (const auto length : history_lengths)
      prefix_cumulative.push_back(prefix_cumulative.back() + int32_t(length));
    const auto values_base = at::randn({tokens, channels + 8}, generator, cpu_options).to(device);
    auto values = values_base.narrow(1, 4, channels);
    auto ab_base = at::randn({tokens, 2 * vh + 3}, generator, cpu_options).to(device);
    auto ab = ab_base.narrow(1, tokens % 2, 2 * vh);
    fill_gates(ab, tokens, vh);
    auto history =
        at::randn({prefix_cumulative.back(), channels}, generator, cpu_options).to(device);
    auto weights = (at::randn({channels, 4}, generator, cpu_options) * .2).contiguous().to(device);
    auto cumulative_tensor = at::tensor(cumulative, options.dtype(at::kInt));
    auto history_cumulative = at::tensor(prefix_cumulative, options.dtype(at::kInt));
    auto sources = at::tensor(source_ids, options.dtype(at::kLong));
    auto a_log = at::linspace(-2.f, 2.f, vh, options.dtype(at::kFloat));
    auto dt_bias = at::linspace(-.5f, .5f, vh, options.dtype(at::kFloat));
    auto convolution = infinity::qwen35_conv_cuda(values,
                                                  weights,
                                                  cumulative_tensor,
                                                  int64_t(lengths.size()),
                                                  sources,
                                                  history,
                                                  history_cumulative);
    auto [qk, decay] =
        infinity::qwen35_gdn_prepare_cuda(convolution, ab.contiguous(), a_log, dt_bias, kh, vh);
    auto qk_base = at::empty({tokens * kh, 264}, options.dtype(at::kFloat));
    auto qk_strided = qk_base.narrow(1, 0, 256);
    qk_strided.copy_(qk);
    auto gate_base = at::empty({tokens, 2 * vh + 3}, options.dtype(at::kFloat));
    auto gate_strided = gate_base.narrow(1, 0, 2 * vh);
    gate_strided.copy_(decay);
    auto expected =
        infinity::flashinfer_gdn_pack_cuda(convolution, qk_strided, gate_strided, kh, vh);
    auto fused = infinity::qwen35_gdn_prepare_pack_cuda(values,
                                                        weights,
                                                        cumulative_tensor,
                                                        ab,
                                                        a_log,
                                                        dt_bias,
                                                        kh,
                                                        vh,
                                                        sources,
                                                        history,
                                                        history_cumulative);
    for (size_t i = 0; i < fused.size(); ++i)
      TORCH_CHECK(fused[i].is_contiguous() &&
                      reinterpret_cast<uintptr_t>(fused[i].const_data_ptr()) % 16 == 0 &&
                      at::equal(fused[i].view(at::kByte), expected[i].view(at::kByte)),
                  "Cached GDN convolution/prepare/pack changed provider buffers kh=",
                  kh,
                  " vh=",
                  vh,
                  " suffix=",
                  tokens,
                  " buffer=",
                  i);
    infinity::PackedInput input;
    input.offsets = cumulative.data();
    input.batch = int64_t(lengths.size());
    input.max_sequence = max_sequence;
    input.cumulative = cumulative_tensor;
    input.block_cache_sources = sources;
    input.block_prefix_sources.assign(source_ids.begin(), source_ids.end());
    auto initial = at::randn({int64_t(history_lengths.size()), vh, 128, 128},
                             generator,
                             at::TensorOptions().dtype(at::kFloat))
                       .to(device);
    auto immutable = initial.clone();
    auto packed_output = at::empty({tokens, vh, 128}, options);
    auto legacy_output = at::empty_like(packed_output);
    at::Tensor packed_state, legacy_state;
    if (infinity::flashinfer_gdn_available(device)) {
      TORCH_CHECK(
          infinity::flashinfer_gdn_resume_packed(
              fused, input, packed_output, kh, vh, &packed_state, initial) &&
              infinity::flashinfer_gdn_resume(
                  convolution, qk, decay, input, legacy_output, kh, vh, &legacy_state, initial) &&
              at::equal(packed_output, legacy_output) && at::equal(packed_state, legacy_state) &&
              at::equal(initial, immutable),
          "Packed GDN resume changed provider output/state or initial state");
    }
    auto z_base = at::randn({tokens, vh * 128 + 8}, generator, cpu_options).to(device);
    auto z = z_base.narrow(1, 4, vh * 128);
    auto norm = at::ones({128}, options);
    auto strided_output = infinity::qwen35_delta_cuda(values,
                                                      z,
                                                      ab.contiguous(),
                                                      weights,
                                                      a_log,
                                                      dt_bias,
                                                      norm,
                                                      input,
                                                      kh,
                                                      vh,
                                                      1e-6,
                                                      initial,
                                                      nullptr,
                                                      history,
                                                      history_cumulative,
                                                      false);
    auto contiguous_output = infinity::qwen35_delta_cuda(values.contiguous(),
                                                         z.contiguous(),
                                                         ab.contiguous(),
                                                         weights,
                                                         a_log,
                                                         dt_bias,
                                                         norm,
                                                         input,
                                                         kh,
                                                         vh,
                                                         1e-6,
                                                         initial,
                                                         nullptr,
                                                         history,
                                                         history_cumulative,
                                                         false);
    TORCH_CHECK(at::equal(strided_output, contiguous_output) && at::equal(initial, immutable),
                "Cached suffix changed outputs for padded QKV/Z rows");
  };
  check_cached(1, 4, {1}, {0}, {0});
  check_cached(1, 6, {30, 27}, {2, 3}, {1, 0});
  check_cached(3, 18, {100, 85}, {2}, {0, 0});
  std::cout << "Cached GDN convolution/prepare/pack matches the old route bit for bit\n";
  std::cout << "Fused GDN convolution/prepare/pack matches the old route bit for bit\n";
  for (int width : {1, 3, 4, 7, 8, 16, 32, 33, 64, 128, 256, 384, 1024, 2048, 4096, 5120, 8192})
    for (int layout : {0, 1, 2}) {
      bool strided = layout == 1;
      int offset = layout == 2;
      auto base = (at::randn({2, 3, width * (strided ? 2 : 1) + offset}) * 3).to(at::kBFloat16);
      if (strided)
        base.narrow(-1, width, width).fill_(NAN);
      auto x = base.narrow(-1, offset, width),
           gamma = (at::randn({width + offset}) * .2).to(at::kBFloat16).narrow(0, offset, width);
      auto cuda_base = base.to(device), cuda_x = cuda_base.narrow(-1, offset, width);
      auto cuda_gamma =
          at::zeros({width + offset}, gamma.options().device(device)).narrow(0, offset, width);
      cuda_gamma.copy_(gamma);
      auto actual = infinity::qwen35_reference_norm_cuda(cuda_x, cuda_gamma, 1e-6).cpu();
      TORCH_CHECK(at::equal(actual, independent_norm(x, gamma, 1e-6)),
                  "Independent original binary-tree norm oracle failed width=",
                  width,
                  " strided=",
                  strided);
      if (!strided) {
        auto residual = cuda_x.reshape({6, width}).contiguous();
        auto addition = (residual.flip({0}) * .25).to(at::kBFloat16);
        addition.select(0, 0).copy_(-residual.select(0, 0));
        auto expected = (residual + addition).cpu();
        auto fused =
            infinity::qwen35_reference_residual_norm_cuda(residual, addition, cuda_gamma, 1e-6);
        TORCH_CHECK(
            at::equal(fused.first.cpu(), expected) &&
                at::equal(fused.second.cpu(), independent_norm(expected, gamma, 1e-6)),
            "Fused residual norm changed BF16 addition or the independent norm oracle width=",
            width,
            " offset=",
            offset);
        auto residual_before = residual.clone(), addition_before = addition.clone();
        std::pair<at::Tensor, at::Tensor> quantized;
        auto with_quantization = infinity::qwen35_reference_residual_norm_cuda(
            residual, addition, cuda_gamma, 1e-6, &quantized);
        auto separate = infinity::fp8_quantize_rows(fused.second);
        TORCH_CHECK(
            at::equal(with_quantization.first, fused.first) &&
                at::equal(with_quantization.second, fused.second) &&
                at::equal(quantized.first.view(at::kByte), separate.first.view(at::kByte)) &&
                at::equal(quantized.second, separate.second) &&
                at::equal(residual, residual_before) && at::equal(addition, addition_before),
            "Fused Qwen norm quantization changed rounded rows, E4M3 bytes, row scales or inputs");
        std::pair<at::Tensor, at::Tensor> omitted_quantized;
        auto omitted = infinity::qwen35_reference_residual_norm_cuda(
            residual, addition, cuda_gamma, 1e-6, &omitted_quantized, false);
        TORCH_CHECK(
            !omitted.second.defined() && at::equal(omitted.first, with_quantization.first) &&
                at::equal(omitted_quantized.first.view(at::kByte),
                          quantized.first.view(at::kByte)) &&
                at::equal(omitted_quantized.second, quantized.second) &&
                at::equal(omitted_quantized.first.view(at::kByte),
                          separate.first.view(at::kByte)) &&
                at::equal(omitted_quantized.second, separate.second) &&
                at::equal(residual, residual_before) && at::equal(addition, addition_before),
            "Omitted Qwen norm output changed updated rows, E4M3 bytes, row scales or inputs");
      }
    }
  // Production hidden widths exercise the dynamic-shared staging path with zero,
  // large-magnitude and BF16-subnormal operands. The unquantized kernel remains the oracle.
  auto check_staged_quantization = [&](int64_t width,
                                       const at::Tensor& residual_cpu,
                                       const at::Tensor& addition_cpu,
                                       const at::Tensor& gamma_cpu,
                                       const char* label) {
    auto cuda_gamma = gamma_cpu.to(device);
    auto residual = residual_cpu.to(device), addition = addition_cpu.to(device);
    auto residual_before = residual.clone(), addition_before = addition.clone();
    auto expected = (residual + addition).cpu();
    auto plain =
        infinity::qwen35_reference_residual_norm_cuda(residual, addition, cuda_gamma, 1e-6);
    TORCH_CHECK(at::equal(plain.first.cpu(), expected) &&
                    at::equal(plain.second.cpu(), independent_norm(expected, gamma_cpu, 1e-6)),
                "Staged-norm reference changed ",
                label,
                " width=",
                width);
    std::pair<at::Tensor, at::Tensor> quantized;
    auto staged = infinity::qwen35_reference_residual_norm_cuda(
        residual, addition, cuda_gamma, 1e-6, &quantized);
    auto separate = infinity::fp8_quantize_rows(plain.second);
    TORCH_CHECK(at::equal(staged.first, plain.first) && at::equal(staged.second, plain.second) &&
                    at::equal(quantized.first.view(at::kByte), separate.first.view(at::kByte)) &&
                    at::equal(quantized.second, separate.second),
                "Shared-staged Qwen norm quantization changed ",
                label,
                " width=",
                width);
    std::pair<at::Tensor, at::Tensor> omitted_quantized;
    auto omitted = infinity::qwen35_reference_residual_norm_cuda(
        residual, addition, cuda_gamma, 1e-6, &omitted_quantized, false);
    TORCH_CHECK(
        !omitted.second.defined() && at::equal(omitted.first, staged.first) &&
            at::equal(omitted_quantized.first.view(at::kByte), quantized.first.view(at::kByte)) &&
            at::equal(omitted_quantized.second, quantized.second) &&
            at::equal(omitted_quantized.first.view(at::kByte), separate.first.view(at::kByte)) &&
            at::equal(omitted_quantized.second, separate.second) &&
            at::equal(residual, residual_before) && at::equal(addition, addition_before),
        "Omitted shared-staged Qwen norm changed ",
        label,
        " width=",
        width);
    TORCH_CHECK(at::equal(residual, residual_before) && at::equal(addition, addition_before),
                "Shared-staged Qwen norm mutated an input ",
                label,
                " width=",
                width);
  };
  for (int64_t width : {2048, 4096, 5120, 8192}) {
    auto gamma = (at::randn({width}) * .2).to(at::kBFloat16);
    auto zero = at::zeros({4, width}, gamma.options());
    auto large = at::full({4, width}, 1e6f, gamma.options());
    auto subnormal = at::full({4, width}, std::ldexp(1.f, -133), gamma.options());
    check_staged_quantization(width, zero, at::zeros_like(zero), gamma, "zeros");
    check_staged_quantization(width, large, large * .25, gamma, "large");
    check_staged_quantization(width, subnormal, subnormal, gamma, "subnormal");
  }
  auto rejects = [&](const at::Tensor& x, const at::Tensor& gamma) {
    bool rejected = false;
    try {
      (void)infinity::qwen35_reference_norm_cuda(x, gamma, 1e-6);
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Invalid norm operands must reject before launch");
  };
  auto opts = at::TensorOptions().device(device).dtype(at::kBFloat16);
  rejects(at::empty({2, 0}, opts), at::empty({0}, opts));
  rejects(at::ones({2, 7}, opts), at::ones({7}, opts.device(at::kCPU)));
  rejects(at::ones({2, 7}, opts), at::ones({1, 7}, opts));
  bool omitted_without_quantized = false;
  try {
    (void)infinity::qwen35_reference_residual_norm_cuda(at::zeros({2, 64}, opts),
                                                        at::zeros({2, 64}, opts),
                                                        at::ones({64}, opts),
                                                        1e-6,
                                                        nullptr,
                                                        false);
  } catch (const c10::Error&) {
    omitted_without_quantized = true;
  }
  TORCH_CHECK(omitted_without_quantized,
              "Omitting normalized output without FP8 storage must reject before launch");
  // Deterministic synthetic operands exercise packed recurrence isolation.
  auto q = at::randn({32, 384}, opts) * .2, z = at::randn({32, 128}, opts) * .2;
  auto ab = at::randn({32, 2}, opts) * .2, conv = at::randn({384, 4}, opts) * .2;
  auto alog = at::zeros({1}, opts.dtype(at::kFloat));
  auto dt = at::zeros({1}, opts.dtype(at::kFloat)), weight = at::ones({128}, opts);
  int32_t offsets[3] = {0, 32, 64};
  infinity::PackedInput input;
  input.offsets = offsets;
  input.batch = 1;
  input.max_sequence = 32;
  input.cumulative =
      at::from_blob(offsets, {2}, at::TensorOptions().dtype(at::kInt)).clone().to(device);
  auto expected =
      infinity::qwen35_reference_delta_cuda(q, z, ab, conv, alog, dt, weight, input, 1, 1, 1e-6);
  TORCH_CHECK(at::isfinite(expected).all().item<bool>(), "Synthetic recurrence must be finite");
  input.batch = 2;
  input.cumulative =
      at::from_blob(offsets, {3}, at::TensorOptions().dtype(at::kInt)).clone().to(device);
  auto twice = infinity::qwen35_reference_delta_cuda(at::cat({q, q}, 0),
                                                     at::cat({z, z}, 0),
                                                     at::cat({ab, ab}, 0),
                                                     conv,
                                                     alog,
                                                     dt,
                                                     weight,
                                                     input,
                                                     1,
                                                     1,
                                                     1e-6);
  TORCH_CHECK(at::equal(twice, at::cat({expected, expected}, 0)),
              "Packed sequence recurrence state leaked");
  auto poisoned = infinity::qwen35_reference_delta_cuda(at::cat({q, at::full_like(q, NAN)}, 0),
                                                        at::cat({z, z}, 0),
                                                        at::cat({ab, ab}, 0),
                                                        conv,
                                                        alog,
                                                        dt,
                                                        weight,
                                                        input,
                                                        1,
                                                        1,
                                                        1e-6);
  TORCH_CHECK(at::equal(poisoned.narrow(0, 0, 32), expected),
              "Independent sequence contaminated by later NaNs");
  for (bool fma : {false, true})
    for (auto geometry : {std::pair<int64_t, int64_t>{1, 1},
                          {2, 4},
                          {4, 8},
                          {16, 16},
                          {16, 32},
                          {16, 48},
                          {32, 64},
                          {32, 96},
                          {16, 128}}) {
      resumed_delta(device, geometry.first, geometry.second, fma);
      independent_recurrent(device, geometry.first, geometry.second, fma);
    }
  for (auto geometry : {std::pair<int64_t, int64_t>{16, 16}, {16, 48}, {32, 128}})
    resumed_delta(device, geometry.first, geometry.second, true, true);
  // A lone 185-token suffix and eight copies must use the same arithmetic.
  // Reuse the cache/state oracle with a real GVA geometry across the old gate.
  resumed_delta(device, 16, 48, true, false, 198);
  std::cout << "Independent original norms and synthetic packed sequence isolation passed\n";
  std::cout << "DeltaNet cache isolation and split/resume passed in both arithmetic modes\n";
}
