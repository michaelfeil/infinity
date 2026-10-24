#pragma once
#include "batching/prefix_cache.h"
#include "models/model.h"

inline void check_complete_classifier_cache(tei::Model& model, const tei::PackedInput& input) {
  struct Observed final : tei::Model {
    tei::Model& inner;
    mutable size_t predictions = 0;

    explicit Observed(tei::Model& model) : inner(model) {}

    void ready() override {}

    bool supports_prefix_cache(int operation) const override {
      return inner.supports_prefix_cache(operation);
    }

    int64_t prefix_layers() const override {
      return inner.prefix_layers();
    }

    uint64_t prefix_payload_bytes(int64_t tokens) const override {
      return inner.prefix_payload_bytes(tokens);
    }

    bool prefix_checkpoint_only() const override {
      return inner.prefix_checkpoint_only();
    }

    at::Tensor forward(const tei::PackedInput& value) const override {
      return inner.forward(value);
    }

    at::Tensor predict(const tei::PackedInput& value, bool tokens) const override {
      ++predictions;
      return inner.predict(value, tokens);
    }

    at::Tensor classify_hidden(const tei::PackedInput& value,
                               const at::Tensor& hidden) const override {
      return inner.classify_hidden(value, hidden);
    }
  } observed(model);

  tei::PrefixCache cache(16 << 20, 4096, 8);
  auto expected = model.predict(input, false);
  TORCH_CHECK(at::equal(cache.run(observed, input, 1), expected), "Cold classifier changed logits");
  const auto bytes = cache.bytes();
  auto second = cache.run(observed, input, 1);
  TORCH_CHECK(at::equal(second, expected), "Complete classifier reuse changed logits");
  TORCH_CHECK(observed.predictions == 1, "Complete classifier input reran the backbone");
  second.zero_();
  TORCH_CHECK(at::equal(cache.run(observed, input, 1), expected),
              "Caller mutation changed cached classifier state");
  TORCH_CHECK(observed.predictions == 1 && cache.bytes() == bytes,
              "Complete classifier reuse allocated another snapshot");
  auto changed = input;
  changed.max_sequence++;
  cache.run(observed, changed, 1);
  TORCH_CHECK(observed.predictions == 2,
              "Complete classifier reuse ignored original batch geometry");
  cache.clear();
  TORCH_CHECK(cache.bytes() == 0, "Classifier cache retained state after clear");
}

inline void check_prefix_cache(tei::Model& model,
                               const tei::PackedInput& input,
                               int operation = 0,
                               double rtol = 1e-5,
                               double atol = 2e-5) {
  tei::PrefixCache cache(16 << 20, 32, 8);
  auto reference = operation ? model.predict(input, operation == 2) : model.forward(input);
  auto first = cache.run(model, input, operation);
  TORCH_CHECK(at::allclose(first, reference, rtol, atol),
              "Cold prefix-cache execution changed model outputs");
  first.zero_(); // Caller ownership must not mutate a resident snapshot.
  auto second = cache.run(model, input, operation);
  TORCH_CHECK(at::allclose(second, reference, rtol, atol),
              "Warm prefix-cache execution changed model outputs, maxabs=",
              (second - reference).abs().max().item<float>());
  if (model.supports_prefix_cache(operation)) {
    TORCH_CHECK(cache.stats().hits > 0, "Eligible model never reused prefix state");
  } else {
    TORCH_CHECK(cache.stats().bypasses == 2 && cache.stats().hits == 0,
                "Ineligible execution reused prefix state");
  }
  TORCH_CHECK(cache.bytes() <= 16 << 20, "Prefix-cache resident budget exceeded");
}
