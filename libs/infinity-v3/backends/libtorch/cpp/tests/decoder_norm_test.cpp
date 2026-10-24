// Independent CUDA checks for residual precision and BF16 normalization.
#include "kernels/decoder_norm.h"
#include "kernels/decoder_packed_readout.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <utility>

namespace {
void verify_subnormal_router(const c10::Device& device) {
  auto host = at::full({1, 128}, -100.f, at::TensorOptions().dtype(at::kFloat));
  auto* logits = host.mutable_data_ptr<float>();
  logits[0] = 0.f;
  for (int rank = 0; rank < 7; ++rank)
    logits[127 - rank] = -90.f - rank;
  auto probabilities = infinity::decoder_half_moe_probabilities(host.to(device));
  auto observed = probabilities.cpu();
  const auto* values = observed.const_data_ptr<float>();
  for (int expert = 121; expert < 128; ++expert)
    TORCH_CHECK(values[expert] > 0.f && values[expert] < std::numeric_limits<float>::min(),
                "Exact MoE softmax must retain subnormal expert probabilities");
  auto ids = at::argsort(probabilities, true, -1, true).narrow(-1, 0, 8).cpu();
  const auto* selected = ids.const_data_ptr<int64_t>();
  TORCH_CHECK(selected[0] == 0, "MoE maximum expert changed");
  for (int rank = 1; rank < 8; ++rank)
    TORCH_CHECK(selected[rank] == 128 - rank,
                "Subnormal router probabilities changed top8 expert order");
}

bool fail_cached_matmul = false;
int cached_matmul_calls = 0;
int64_t expected_live_bytes_at_fallback = -1;

template <class... Args> cublasStatus_t probe_cached_matmul(Args... args) {
  ++cached_matmul_calls;
  if (std::exchange(fail_cached_matmul, false))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  return ::cublasLtMatmul(args...);
}
}

namespace at {
Tensor probe_cache_linear(const Tensor& x, const Tensor& weight) {
  if (expected_live_bytes_at_fallback >= 0)
    TORCH_CHECK(c10::cuda::CUDACachingAllocator::getDeviceStats(x.get_device())
                        .allocated_bytes[0]
                        .current == expected_live_bytes_at_fallback,
                "Fallback entry must not retain a speculative result allocation");
  return at::linear(x, weight);
}
}

// Give the fault-injected copy private symbols; production cache calls remain
// unchanged, including ATen's independent fallback implementation.
#define CachedNoBiasLt DecoderCacheUnderTest
#define cublasLtMatmul probe_cached_matmul
#define linear probe_cache_linear
#include "kernels/cached_decoder_linear.h"
#undef linear
#undef cublasLtMatmul
#undef CachedNoBiasLt

int main(int argc, char** argv) {
  c10::InferenceMode inference;
  at::manual_seed(43);
  auto options =
      at::TensorOptions().dtype(at::kBFloat16).device(c10::Device(argc > 1 ? argv[1] : "cuda:0"));
  verify_subnormal_router(options.device());
  // Small aligned chunks execute the retained Candle adapter, independently
  // checking the specialized prefill kernel at its dispatch boundary.
  for (int64_t rows : {127, 128, 257})
    for (bool residual : {false, true}) {
      auto x = at::randn({rows, 4096}, options), r = at::randn_like(x),
           w = at::randn({4096}, options);
      for (double eps : {1e-6, 1e-5}) {
        std::vector<at::Tensor> normalized, summed;
        for (int64_t start = 0; start < rows; start += 64) {
          auto count = std::min(int64_t(64), rows - start);
          auto part =
              infinity::decoder_exact_rms_norm(x.narrow(0, start, count),
                                               residual ? r.narrow(0, start, count) : at::Tensor(),
                                               w,
                                               eps);
          normalized.push_back(part.first);
          summed.push_back(part.second);
        }
        auto actual = infinity::decoder_exact_rms_norm(x, residual ? r : at::Tensor(), w, eps);
        TORCH_CHECK(at::equal(actual.first, at::cat(normalized)),
                    "BF16 prefill RMS changed adapter output");
        TORCH_CHECK(at::equal(actual.second, at::cat(summed)), "BF16 prefill RMS changed residual");
        if (!residual)
          TORCH_CHECK(actual.second.is_alias_of(x), "BF16 prefill RMS input alias");
      }
    }
  for (int64_t width : {896, 1024, 2048, 3072, 4096}) {
    for (int64_t rows : {1, 7, 257}) {
      auto x = (at::rand({rows, width}, options) - .5), r = (at::rand_like(x) - .5),
           w = at::ones({width}, options);
      for (bool residual : {false, true})
        for (double eps : {1e-6, 1e-3}) {
          auto result = infinity::decoder_exact_rms_norm(x, residual ? r : at::Tensor(), w, eps);
          auto unrounded = residual ? x.to(at::kFloat) + r.to(at::kFloat) : x.to(at::kFloat);
          auto expected =
              (unrounded * at::rsqrt(unrounded.square().mean(-1, true) + eps)).to(at::kBFloat16);
          TORCH_CHECK(
              (result.first - expected).abs().max().item<float>() <= .015625,
              "Exact RMS differs by more than one BF16 output quantum from independent FP32 statistics");
          TORCH_CHECK(at::equal(result.second, residual ? x + r : x),
                      "RMS residual must round only at its output store");
          if (!residual)
            TORCH_CHECK(result.second.is_alias_of(x),
                        "Plain RMS must retain the input residual without a copy");
        }
    }
    auto x = at::ones({1, width}, options);
    x.narrow(1, width / 2, width / 2).fill_(2);
    auto r = at::full_like(x, .005), w = at::full({width}, 64., options);
    auto result = infinity::decoder_exact_rms_norm(x, r, w, 1e-6);
    auto full = x.to(at::kFloat) + r.to(at::kFloat);
    auto expected = (full * at::rsqrt(full.square().mean(-1, true) + 1e-6) * w.to(at::kFloat))
                        .to(at::kBFloat16);
    auto rounded = (x + r).to(at::kFloat);
    auto wrong = (rounded * at::rsqrt(rounded.square().mean(-1, true) + 1e-6) * w.to(at::kFloat))
                     .to(at::kBFloat16);
    TORCH_CHECK(at::equal(result.first, expected),
                "Known residual-rounding regression must match independent FP32 oracle exactly");
    TORCH_CHECK((result.first - wrong).abs().max().item<float>() >= .125,
                "Fixture must distinguish pre-rounded residual statistics");
    bool rejected = false;
    try {
      infinity::decoder_exact_rms_norm(x.to(at::kHalf), r.to(at::kHalf), w.to(at::kHalf), 1e-6);
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected == (width != 1024 && width != 896 && width != 2048),
                "FP16 RMS must accept only registered widths 896, 1024 and 2048");
  }
  // FP16 uses its original registered 1024-wide geometry and preserves
  // FP32 residual statistics before the output store.
  auto half_options = options.dtype(at::kHalf);
  for (int64_t width : {896, 1024, 2048})
    for (int64_t rows : {1, 7, 257})
      for (bool residual : {false, true}) {
        auto x = at::rand({rows, width}, half_options) - .5, r = at::rand_like(x) - .5;
        auto w = at::ones({width}, half_options);
        auto result = infinity::decoder_exact_rms_norm(x, residual ? r : at::Tensor(), w, 1e-6);
        auto full = residual ? x.to(at::kFloat) + r.to(at::kFloat) : x.to(at::kFloat);
        auto expected = (full * at::rsqrt(full.square().mean(-1, true) + 1e-6)).to(at::kHalf);
        TORCH_CHECK((result.first - expected).abs().max().item<float>() <= .001953125,
                    "FP16 RMS must preserve independent FP32 residual statistics");
        TORCH_CHECK(at::equal(result.second, residual ? x + r : x),
                    "FP16 residual must round at its output store");
        if (!residual)
          TORCH_CHECK(result.second.is_alias_of(x),
                      "Plain FP16 RMS must retain the input residual");
      }
  bool rejected_width = false;
  try {
    auto x = at::ones({1, 512}, options);
    infinity::decoder_exact_rms_norm(x, {}, at::ones({512}, options), 1e-6);
  } catch (const c10::Error&) {
    rejected_width = true;
  }
  TORCH_CHECK(rejected_width, "Unregistered width must be rejected");
  // Independent packed readout oracle: interior/trailing PAD, all-PAD first
  // fallback, then changing actual IDs without changing cumulative layout.
  for (int64_t width : {16, 129, 896, 1024, 2048}) {
    constexpr int64_t pad = 31;
    auto hidden =
        at::arange(10 * width, options.dtype(at::kFloat)).view({10, width}).to(at::kBFloat16);
    auto ids = at::tensor(at::IntArrayRef{5, pad, 6, pad, pad, pad, pad, 9, pad, 11},
                          options.dtype(at::kLong));
    auto cu = at::tensor({0, 4, 6, 10}, options.dtype(at::kInt));
    auto expected = hidden.index_select(0, at::tensor({2, 4, 9}, options.dtype(at::kLong)));
    auto selected = infinity::decoder_packed_classifier_rows(hidden, ids, cu, 3, pad);
    TORCH_CHECK(at::equal(selected, expected),
                "Packed classifier readout changed selected hidden bits");
    ids.copy_(at::tensor(at::IntArrayRef{5, pad, 6, 7, pad, 7, pad, 9, pad, pad},
                         options.dtype(at::kLong)));
    expected = hidden.index_select(0, at::tensor({3, 5, 7}, options.dtype(at::kLong)));
    TORCH_CHECK(
        at::equal(infinity::decoder_packed_classifier_rows(hidden, ids, cu, 3, pad), expected),
        "Packed readout reused stale ID selection");
    auto half_hidden = hidden.to(at::kHalf);
    TORCH_CHECK(
        at::equal(infinity::decoder_packed_classifier_rows(half_hidden, ids, cu, 3, pad),
                  infinity::decoder_packed_classifier_rows(hidden, ids, cu, 3, pad).to(at::kHalf)),
        "Packed FP16 gather must preserve every original bit");
    auto last = hidden.index_select(0, at::tensor({3, 5, 9}, options.dtype(at::kLong)));
    TORCH_CHECK(
        at::equal(infinity::decoder_packed_classifier_rows(hidden, ids, cu, 3, pad, false), last),
        "Absent PAD configuration changed end-row selection");
    auto bits = (at::arange(10 * width, options.dtype(at::kInt)) * 73)
                    .remainder(65536)
                    .to(at::kShort)
                    .view({10, width});
    auto indices = at::tensor({3, 5, 7}, options.dtype(at::kLong));
    for (auto type : {at::kHalf, at::kBFloat16}) {
      auto rows = infinity::decoder_packed_classifier_rows(bits.view(type), ids, cu, 3, pad);
      TORCH_CHECK(at::equal(rows.view(at::kShort), bits.index_select(0, indices)),
                  "Packed gather changed raw16-bit payloads");
    }
    auto float_bits = (at::arange(10 * width, options.dtype(at::kLong)) * 2654435761LL)
                          .to(at::kInt)
                          .view({10, width});
    auto float_rows =
        infinity::decoder_packed_classifier_rows(float_bits.view(at::kFloat), ids, cu, 3, pad);
    TORCH_CHECK(at::equal(float_rows.view(at::kInt), float_bits.index_select(0, indices)),
                "Packed gather changed raw32-bit payloads");
    bool rejected = false;
    try {
      infinity::decoder_packed_classifier_rows(hidden.to(at::kDouble), ids, cu, 3, pad);
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Unvalidated packed readout dtype must be rejected");
  }
  // Exercise recoverable execution failure and capacity fallback on the
  // cache's qualified hardware, without changing the production operator.
  if (at::cuda::getCurrentDeviceProperties()->major == 9 &&
      at::cuda::getCurrentDeviceProperties()->minor == 0) {
    const auto prior = at::globalContext().blasPreferredBackend();
    at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublaslt);
    infinity::DecoderCacheUnderTest cache;
    auto weight = at::randn({64, 896}, options) * .03, x = at::randn({8, 896}, options) * .03;
    auto expected = at::linear(x, weight);
    auto first = cache.forward(x, weight);
    TORCH_CHECK(cached_matmul_calls > 0 && at::allclose(first, expected, .015625, .001),
                "Fixture must execute a valid cached algorithm");
    fail_cached_matmul = true;
    auto recovered = cache.forward(x, weight);
    TORCH_CHECK(!fail_cached_matmul && at::equal(recovered, expected),
                "Rejected algorithm must recover through the independent ATen fallback");
    const auto calls = cached_matmul_calls;
    TORCH_CHECK(at::equal(cache.forward(x, weight), expected) && cached_matmul_calls == calls,
                "Rejected cached plan must not be retried");
    infinity::DecoderCacheUnderTest full;
    for (int64_t rows = 1; rows <= 64; ++rows)
      full.forward(at::randn({rows, 896}, options), weight);
    auto unseen = at::randn({65, 896}, options);
    auto allocation = [&] {
      return c10::cuda::CUDACachingAllocator::getDeviceStats(unseen.get_device())
          .allocated_bytes[0];
    };
    auto control = at::linear(unseen, weight);
    expected_live_bytes_at_fallback = allocation().current;
    const auto saturated_calls = cached_matmul_calls;
    auto fallback = full.forward(unseen, weight);
    expected_live_bytes_at_fallback = -1;
    TORCH_CHECK(cached_matmul_calls == saturated_calls && at::equal(fallback, control),
                "Full cache must preserve independent ATen fallback output");
    if (cublasLtGetVersion() == 130101) {
      auto stream = c10::cuda::getStreamFromPool(false, options.device().index());
      c10::cuda::CUDAStreamGuard stream_guard(stream);
      for (int width : {24576, 28672}) {
        infinity::DecoderCacheUnderTest prefill;
        auto input = at::randn({2048, 4096}, options),
             projection = at::randn({width, 4096}, options) * .02;
        auto reference = at::linear(input, projection);
        const auto before = cached_matmul_calls;
        auto result = prefill.forward(input, projection, true);
        TORCH_CHECK(cached_matmul_calls > before && at::equal(result, reference),
                    "Qualified BF16 prefill tile must execute and match ATen");
        stream.synchronize();
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        result = prefill.forward(input, projection, true);
        graph.capture_end();
        input.add_(.25);
        graph.replay();
        TORCH_CHECK(at::equal(result, at::linear(input, projection)),
                    "Cached prefill graph must read refreshed input on the current stream");
        fail_cached_matmul = true;
        auto recovered = prefill.forward(input, projection, true);
        const auto retired = cached_matmul_calls;
        TORCH_CHECK(!fail_cached_matmul && at::equal(recovered, at::linear(input, projection)),
                    "Prefill execution rejection must fall back");
        prefill.forward(input, projection, true);
        TORCH_CHECK(cached_matmul_calls == retired, "Retired prefill tile must not be retried");
      }
    }
    at::globalContext().setBlasPreferredBackend(prior);
  }
  std::cout << "Exact decoder RMS registered widths 896/1024/2048/3072/4096 tests passed\n";
}
