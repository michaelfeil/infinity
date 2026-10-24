// Minimal native harness for independently generated Transformers fixtures.
#include "../cpp/models/encoder_models.h"
#include <c10/core/InferenceMode.h>
#include <cstring>

namespace {
struct Fixture {
  tei::Options options;
  tei::Weights weights;
  std::unique_ptr<tei::Model> model;
};

thread_local std::string error;

template <class F> int checked(F fn) {
  try {
    c10::InferenceMode guard;
    fn();
    return 0;
  } catch (const std::exception& e) {
    error = e.what();
    return -1;
  }
}
}

extern "C" {
const char* fixture_error() {
  return error.c_str();
}

void* fixture_create() {
  return new Fixture;
}

void fixture_delete(void* p) {
  delete static_cast<Fixture*>(p);
}

void fixture_option(void* p, const char* k, const char* v) {
  static_cast<Fixture*>(p)->options.values[k] = v;
}

int fixture_weight(void* p, const char* k, const float* data, const int64_t* shape, int64_t rank) {
  return checked([&] {
    static_cast<Fixture*>(p)->weights[k] = at::from_blob(const_cast<float*>(data),
                                                         at::IntArrayRef(shape, rank),
                                                         at::TensorOptions().dtype(at::kFloat))
                                               .clone();
  });
}

int fixture_ready(void* p, int gpu) {
  return checked([&] {
    auto& f = *static_cast<Fixture*>(p);
    c10::Device device(gpu ? "cuda:2" : "cpu");
    auto dtype = gpu ? at::kHalf : at::kFloat;
    for (auto& [k, w] : f.weights)
      w = w.to(device, dtype);
    f.model = tei::create_encoder(f.options, f.weights, device, dtype);
    TORCH_CHECK(f.model, "Unsupported encoder fixture");
    f.model->ready();
  });
}

int fixture_forward(void* p,
                    const int64_t* ids,
                    const int64_t* types,
                    const int64_t* positions,
                    const int32_t* offsets,
                    int64_t batch,
                    int64_t maximum,
                    int gpu,
                    int prediction,
                    float* output) {
  return checked([&] {
    auto& f = *static_cast<Fixture*>(p);
    auto options = at::TensorOptions().device(at::kCPU).dtype(at::kLong);
    c10::Device device(gpu ? "cuda:2" : "cpu");
    const int64_t tokens = offsets[batch];
    auto tensor = [&](const int64_t* v) {
      return at::from_blob(const_cast<int64_t*>(v), {tokens}, options).to(device);
    };
    auto cu = at::from_blob(const_cast<int32_t*>(offsets), {batch + 1}, options.dtype(at::kInt))
                  .to(device);
    tei::PackedInput in{tensor(ids), tensor(types), tensor(positions), cu, offsets, batch, maximum};
    auto result = prediction ? f.model->predict(in, prediction == 2) : f.model->forward(in);
    result = result.to(at::kCPU, at::kFloat).contiguous();
    std::memcpy(output, result.data_ptr(), result.nbytes());
  });
}
}
