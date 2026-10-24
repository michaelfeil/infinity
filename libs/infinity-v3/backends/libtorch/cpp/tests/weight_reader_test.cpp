#include "helpers/weight_reader.h"
#include <ATen/ATen.h>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <unistd.h>
#include <vector>

int main(int argc, char**) {
  try {
    at::Device device = argc > 1 ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
    char path[] = "/tmp/infinity-weight-test-XXXXXX";
    int fd = mkstemp(path);
    TORCH_CHECK(fd >= 0, "mkstemp failed");
    unlink(path);
    constexpr uint64_t header = 13;
    std::vector<uint8_t> bytes(header + (26 << 20) + 6);
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = (i * 29 + i / 4096) % 251;
    TORCH_CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()),
                "write failed");
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<TeiWeight> weights;
    names.reserve(516);
    shapes.reserve(516);
    weights.reserve(516);
    uint64_t offset = 0;
    auto add = [&](const char* name, int dtype, std::vector<int64_t> shape, size_t length) {
      names.emplace_back(name);
      shapes.push_back(std::move(shape));
      weights.push_back({names.back().c_str(),
                         shapes.back().data(),
                         shapes.back().size(),
                         dtype,
                         offset,
                         length});
      offset += length;
    };
    add("bf16", 2, {1}, 2);
    add("unaligned_f32_scalar", 0, {}, 4);
    add("empty", 1, {0}, 0);
    add("large", 0, {6 << 20}, 24 << 20);
    for (int i = 0; i < 512; ++i) {
      auto name = "small_" + std::to_string(i);
      add(name.c_str(), 1, {2048}, 4096);
    }
    TORCH_CHECK(offset + header == bytes.size());
    int original_flags = fcntl(fd, F_GETFL);
    for (bool direct : {false, true}) {
      std::map<std::string, at::Tensor> result;
      infinity::read_safetensors(fd,
                                 header,
                                 offset,
                                 weights.data(),
                                 weights.size(),
                                 device,
                                 4,
                                 1 << 20,
                                 direct,
                                 [&](const char* name, at::Tensor tensor) {
                                   TORCH_CHECK(result.emplace(name, std::move(tensor)).second);
                                 });
      for (const auto& weight : weights) {
        const auto& tensor = result.at(weight.name);
        TORCH_CHECK(tensor.device() == device && tensor.nbytes() == weight.length);
        // A small weight must not keep a whole shard or staging chunk alive.
        TORCH_CHECK(tensor.storage().nbytes() == weight.length);
        auto host = tensor.flatten().view(at::kByte).to(at::kCPU);
        if (weight.length)
          TORCH_CHECK(std::memcmp(host.data_ptr(),
                                  bytes.data() + header + weight.offset,
                                  weight.length) == 0);
      }
      TORCH_CHECK(fcntl(fd, F_GETFL) == original_flags, "Borrowed descriptor flags changed");
    }
    auto invalid = weights.front();
    invalid.length = 1;
    bool rejected = false;
    try {
      infinity::read_safetensors(
          fd, header, offset, &invalid, 1, device, 4, 65536, true, [](const char*, at::Tensor) {});
    } catch (const std::exception&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Invalid byte length was accepted");
    rejected = false;
    std::map<std::string, at::Tensor> partial;
    try {
      infinity::read_safetensors(fd,
                                 header,
                                 offset,
                                 weights.data(),
                                 weights.size(),
                                 device,
                                 4,
                                 1 << 20,
                                 true,
                                 [&](const char* name, at::Tensor tensor) {
                                   partial.emplace(name, std::move(tensor));
                                   throw std::runtime_error("injected insertion error");
                                 });
    } catch (const std::exception&) {
      rejected = true;
    }
    TORCH_CHECK(rejected && partial.size() == 1);
    TORCH_CHECK(fcntl(fd, F_GETFL) == original_flags, "Error path changed descriptor flags");
    auto first_bytes = partial.begin()->second.flatten().view(at::kByte).to(at::kCPU);
    TORCH_CHECK(std::memcmp(first_bytes.data_ptr(), bytes.data() + header, 2) == 0);

    // A large F32 -> F16/BF16 conversion must work with a 1 MiB staging
    // budget, without a second full-sized source allocation on the GPU.
    std::vector<float> finite(6 << 20);
    for (size_t i = 0; i < finite.size(); ++i)
      finite[i] = (static_cast<int>(i % 511) - 255) / 257.f;
    uint64_t source_bytes = finite.size() * sizeof(float);
    TORCH_CHECK(ftruncate(fd, header + source_bytes) == 0);
    TORCH_CHECK(pwrite(fd, finite.data(), source_bytes, header) ==
                static_cast<ssize_t>(source_bytes));
    int64_t size = finite.size();
    TeiWeight convert{"convert", &size, 1, 0, 0, source_bytes};
    for (auto type : {at::kHalf, at::kBFloat16}) {
      at::Tensor result;
      infinity::read_safetensors(
          fd,
          header,
          source_bytes,
          &convert,
          1,
          device,
          4,
          1 << 20,
          false,
          [&](const char*, at::Tensor tensor) { result = std::move(tensor); },
          [&](const char*, at::ScalarType) { return type; });
      auto expected =
          at::from_blob(finite.data(), {size}, at::TensorOptions().dtype(at::kFloat)).to(type);
      TORCH_CHECK(result.scalar_type() == type && result.storage().nbytes() == finite.size() * 2);
      TORCH_CHECK(at::equal(result.to(at::kCPU), expected), "Dtype conversion changed values");
    }
    close(fd);
    std::cout
        << "weight reader dtype, ownership, bounded staging and direct/buffered checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
