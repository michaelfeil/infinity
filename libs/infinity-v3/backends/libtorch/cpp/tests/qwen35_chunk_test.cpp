// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "kernels/qwen35_chunk.h"
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
  const c10::Device device(argc > 1 ? argv[1] : "cuda:0");
  at::manual_seed(9183);
  constexpr int kh = 2, vh = 4, d = 128, channels = (2 * kh + vh) * d;
  for (bool ragged : {false, true}) {
    std::vector<int32_t> offsets{0};
    for (int n : ragged ? std::vector<int>{1, 31, 32, 33, 63, 64, 65, 257} : std::vector<int>{257})
      offsets.push_back(offsets.back() + n);
    const int tokens = offsets.back(), sequences = offsets.size() - 1;
    auto values = (at::randn({tokens, channels}) * .2).to(at::kBFloat16);
    auto qk = at::randn({tokens, kh, 2, d});
    qk = qk * at::rsqrt(qk.square().sum(-1, true) + 1e-6);
    qk.select(2, 0).mul_(1. / std::sqrt(double(d)));
    qk = qk.reshape({tokens * kh, 2 * d}).contiguous();
    auto gates = at::rand({tokens, 2 * vh}) * .2 + .7;
    gates.narrow(1, vh, vh).sub_(.3);
    // Resets at and around chunk boundaries must not cause log(0)/NaN.
    for (int t : {0, 31, 32, 63, 64, 255, 256})
      gates.narrow(0, t, 1).narrow(1, 0, vh).zero_();
    auto initial = at::randn({3, vh, d, d}) * .03;
    std::vector<int64_t> sources =
        ragged ? std::vector<int64_t>{2, 0, 2, 1, 0, 2, 1, 0} : std::vector<int64_t>{2};
    for (bool cached : {false, true}) {
      auto expected = at::empty({tokens, vh, d}, at::TensorOptions().dtype(at::kFloat));
      auto final = at::empty({sequences, vh, d, d}, at::TensorOptions().dtype(at::kDouble));
      auto vals = values.to(at::kDouble), q = qk.to(at::kDouble).reshape({tokens, kh, 2, d}),
           gate = gates.to(at::kDouble);
      for (int seq = 0; seq < sequences; ++seq) {
        auto state = cached ? initial.select(0, sources[seq]).to(at::kDouble)
                            : at::zeros({vh, d, d}, at::TensorOptions().dtype(at::kDouble));
        for (int t = offsets[seq]; t < offsets[seq + 1]; ++t) {
          auto keys = at::repeat_interleave(q.select(0, t).select(1, 1), vh / kh, 0);
          auto queries = at::repeat_interleave(q.select(0, t).select(1, 0), vh / kh, 0);
          auto v = vals.select(0, t).narrow(0, 2 * kh * d, vh * d).reshape({vh, d});
          state = state * gate.select(0, t).narrow(0, 0, vh).reshape({vh, 1, 1});
          auto delta = (v - (state * keys.unsqueeze(-1)).sum(1)) *
                       gate.select(0, t).narrow(0, vh, vh).unsqueeze(-1);
          state = state + keys.unsqueeze(-1) * delta.unsqueeze(1);
          expected.select(0, t).copy_((state * queries.unsqueeze(-1)).sum(1));
        }
        final.select(0, seq).copy_(state);
      }
      tei::PackedInput input;
      input.offsets = offsets.data();
      input.batch = sequences;
      input.max_sequence = 257;
      input.cumulative = at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
      input.block_prefix_sources = sources;
      input.block_cache_sources =
          at::tensor(sources, at::TensorOptions().dtype(at::kLong)).to(device);
      auto actual = at::empty({tokens, vh, d}, values.options().device(device));
      auto device_values = values.to(device), device_qk = qk.to(device),
           device_gates = gates.to(device), device_initial = initial.to(device);
      auto immutable = device_initial.clone();
      at::Tensor actual_state;
      tei::qwen35_chunk_delta_cuda(device_values,
                                   device_qk,
                                   device_gates,
                                   input,
                                   actual,
                                   kh,
                                   vh,
                                   cached ? device_initial : at::Tensor(),
                                   &actual_state);
      auto host = actual.cpu().to(at::kFloat), oracle = expected.to(at::kBFloat16).to(at::kFloat),
           host_state = actual_state.cpu().to(at::kDouble);
      TORCH_CHECK(at::isfinite(host).all().item<bool>() &&
                      at::isfinite(host_state).all().item<bool>(),
                  "Chunk reset produced nonfinite values");
      TORCH_CHECK(at::allclose(host, oracle, .01, 2e-5),
                  "Chunk output differs from independent FP64 recurrence, max error=",
                  (host - oracle).abs().max().item<double>());
      TORCH_CHECK(at::allclose(host_state, final, 1e-4, 2e-5),
                  "Chunk state differs from FP64 recurrence, max error=",
                  (host_state - final).abs().max().item<double>());
      TORCH_CHECK(at::equal(device_initial, immutable), "Suffix mutated shared initial state");
      std::cout << "Packed chunk recurrence cached=" << cached << " tokens=" << tokens
                << " max_output_error=" << (host - oracle).abs().max().item<double>()
                << " max_state_error=" << (host_state - final).abs().max().item<double>() << "\n";
      // Reject invalid state sources before launching any GPU work.
      input.block_prefix_sources[0] = 3;
      bool rejected = false;
      try {
        tei::qwen35_chunk_delta_cuda(
            device_values, device_qk, device_gates, input, actual, kh, vh, device_initial, nullptr);
      } catch (const c10::Error&) {
        rejected = true;
      }
      TORCH_CHECK(rejected, "Out-of-range prefix state source accepted");
    }
  }
}
