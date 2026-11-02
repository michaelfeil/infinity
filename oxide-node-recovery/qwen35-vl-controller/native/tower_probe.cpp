// Native Qwen3.5 vision tower capture for paired Root qualification.
//
// Build against the public Native sources in this repository (qwen3_vl_models.cpp
// and qwen35_models.cpp) plus the pinned SDK's ATen. It reads a tensor dump
// produced by dump_vision_tensors.py, constructs the real Qwen3VL tower-only
// object with the original Options, and writes merged visual rows as F32.
//
// This probe does not reimplement the tower; it calls Qwen3VL::image_features.

#include "helpers/options.h"
#include "models/qwen3_vl_models.h"
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<int64_t> parse_shape(const std::string& text) {
  std::vector<int64_t> shape;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) if (!item.empty()) shape.push_back(std::stoll(item));
  return shape;
}
std::vector<char> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  TORCH_CHECK(stream, "cannot open ", path);
  stream.seekg(0, std::ios::end);
  const auto size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  std::vector<char> bytes(static_cast<size_t>(size));
  stream.read(bytes.data(), size);
  TORCH_CHECK(stream, "short read ", path);
  return bytes;
}
std::vector<float> read_f32(const std::string& path, size_t count) {
  auto bytes = read_file(path);
  TORCH_CHECK(bytes.size() == count * sizeof(float), "wrong pixel byte count for ", path);
  std::vector<float> values(count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
}  // namespace

int main(int argc, char** argv) {
  TORCH_CHECK(argc == 6,
              "usage: tower_probe TENSOR_ROOT FIXTURES OUT_DIR DTYPE DEVICE");
  const std::string root = argv[1], fixtures_path = argv[2], out_dir = argv[3];
  const std::string dtype_name = argv[4], device_name = argv[5];
  const auto device = c10::Device(device_name);
  const auto dtype = dtype_name == "float16" ? at::kHalf
                    : dtype_name == "float32" ? at::kFloat : at::kBFloat16;
  infinity::Options options;
  {
    std::ifstream stream(root + "/options.txt");
    std::string line;
    while (std::getline(stream, line)) {
      const auto tab = line.find('\t');
      if (tab != std::string::npos) options.values[line.substr(0, tab)] = line.substr(tab + 1);
    }
  }
  auto blob = read_file(root + "/tensors.bin");
  infinity::Weights weights;
  {
    std::ifstream stream(root + "/tensors.txt");
    std::string line;
    while (std::getline(stream, line)) {
      std::stringstream fields(line);
      std::string name, shape_text, offset_text, bytes_text;
      std::getline(fields, name, '\t');
      std::getline(fields, shape_text, '\t');
      std::getline(fields, offset_text, '\t');
      std::getline(fields, bytes_text, '\t');
      const auto shape = parse_shape(shape_text);
      const auto offset = std::stoull(offset_text);
      const auto bytes = std::stoull(bytes_text);
      TORCH_CHECK(offset + bytes <= blob.size(), "tensor range exceeds dump");
      auto tensor = at::from_blob(blob.data() + offset, shape,
                                  at::TensorOptions().dtype(dtype)).clone();
      weights.emplace(name, tensor.to(device));
    }
  }
  c10::InferenceMode inference(true);
  infinity::Qwen3VL tower(options, weights, device, dtype, /*only_vision=*/true);
  tower.ready();
  struct Fixture {
    std::string name, pixels;
    int64_t rows, patch_dim, t, h, w, merge, token_start;
  };
  std::vector<Fixture> fixtures;
  {
    std::ifstream stream(fixtures_path);
    TORCH_CHECK(stream, "cannot open fixtures ", fixtures_path);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream fields(line);
      Fixture fixture;
      std::string rows, patch_dim, t, h, w, merge, token_start;
      std::getline(fields, fixture.name, '\t');
      std::getline(fields, rows, '\t');
      std::getline(fields, patch_dim, '\t');
      std::getline(fields, t, '\t');
      std::getline(fields, h, '\t');
      std::getline(fields, w, '\t');
      std::getline(fields, merge, '\t');
      std::getline(fields, token_start, '\t');
      std::getline(fields, fixture.pixels, '\t');
      fixture.rows = std::stoll(rows); fixture.patch_dim = std::stoll(patch_dim);
      fixture.t = std::stoll(t); fixture.h = std::stoll(h); fixture.w = std::stoll(w);
      fixture.merge = std::stoll(merge); fixture.token_start = std::stoll(token_start);
      fixtures.push_back(std::move(fixture));
    }
  }
  std::vector<infinity::ImageInput> images;
  for (const auto& fixture : fixtures) {
    const auto count = static_cast<size_t>(fixture.rows * fixture.patch_dim);
    auto values = read_f32(fixture.pixels, count);
    auto pixels = at::from_blob(values.data(), {fixture.rows, fixture.patch_dim},
                                at::TensorOptions().dtype(at::kFloat)).clone();
    infinity::ImageInput image;
    image.pixels = pixels;
    image.grid_thw = {fixture.t, fixture.h, fixture.w};
    image.merge_size = fixture.merge;
    image.token_start = fixture.token_start;
    image.token_count = fixture.t * fixture.h * fixture.w / (fixture.merge * fixture.merge);
    images.push_back(std::move(image));
  }
  auto features = tower.image_features(images);
  TORCH_CHECK(features.size() == images.size(), "tower feature count mismatch");
  std::ofstream report(out_dir + "/gold.json");
  report << "{\n  \"arm\": \"native-b66-tower\",\n  \"rows\": [\n";
  for (size_t index = 0; index < features.size(); ++index) {
    auto values = features[index].to(at::kFloat).contiguous().cpu();
    const auto merged = features[index].size(0), hidden = features[index].size(1);
    const auto path = out_dir + "/" + fixtures[index].name + ".f32";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.const_data_ptr<float>()),
                 values.numel() * sizeof(float));
    report << "    {\"name\": \"" << fixtures[index].name << "\", \"merged_rows\": " << merged
           << ", \"out_hidden\": " << hidden << ", \"path\": \"" << path << "\"}"
           << (index + 1 == features.size() ? "\n" : ",\n");
  }
  report << "  ]\n}\n";
  std::cout << "native tower captured " << features.size() << " fixtures\n";
  return 0;
}
