// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//
// SOURCE-ONLY Root packet implementation for Mistral3 / Pixtral.
//
// NON-CLAIM: this file was not compiled, linked, loaded, or executed during the
// CPU-only control. It is staged for Root to build against the frozen b66
// libtei_torch.so inside the GPU/Torch image. It refuses to run unless Root
// overrides the empty CUDA_VISIBLE_DEVICES guard with a real device selection.
//
// The runner implements:
//   create -> flatten original config to tei_option -> load every shard through
//   tei_load_safetensors -> tei_ready -> synchronous/deferred image+text
//   reference forwards -> timing/perf -> tei_destroy.
//
// It uses the exact staged bridge.h C ABI.
// and the packet ABI from ../include/mistral3_b66.h.

#include "../include/mistral3_b66.h"
#include "bridge.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Opaque output lease referenced by the public packet typedef. The engine
// state pointer is a type-erased back-reference for timing counters.
struct mistral3_b66_output_lease {
  void* native = nullptr;
  void* engine_state = nullptr;
  size_t elements = 0;
};

namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

thread_local std::string g_last_error;
thread_local int32_t g_last_error_kind = 0;

int32_t fail(const std::string& message, int32_t kind = 1) {
  g_last_error = message;
  g_last_error_kind = kind;
  return -1;
}

int32_t capture_native(int32_t status) {
  if (status == 0)
    return 0;
  const char* text = tei_error();
  g_last_error = text ? text : "native call failed";
  g_last_error_kind = tei_last_error_kind();
  return status < 0 ? status : -1;
}

template <typename Fn>
int32_t guarded(Fn&& fn) {
  try {
    return fn();
  } catch (const std::exception& error) {
    return fail(error.what());
  } catch (...) {
    return fail("unknown C++ exception");
  }
}

// ---------------------------------------------------------------------------
// Minimal JSON reader used only for config.json, the index, and safetensors
// headers. It keeps the packet free of third-party dependencies.
// ---------------------------------------------------------------------------
struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  const Json* find(const std::string& key) const {
    if (type != Type::Object)
      return nullptr;
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  bool parse(Json& out, std::string& error) {
    skip_space();
    if (!value(out)) {
      error = error_.empty() ? "invalid JSON" : error_;
      return false;
    }
    skip_space();
    if (pos_ != text_.size()) {
      error = "trailing data after JSON value";
      return false;
    }
    return true;
  }

 private:
  void skip_space() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        ++pos_;
      else
        break;
    }
  }

  bool value(Json& out) {
    skip_space();
    if (pos_ >= text_.size())
      return set_error("unexpected end of JSON");
    const char c = text_[pos_];
    if (c == '{')
      return object_value(out);
    if (c == '[')
      return array_value(out);
    if (c == '"') {
      out.type = Json::Type::String;
      return string_value(out.string);
    }
    if (c == 't' || c == 'f')
      return bool_value(out);
    if (c == 'n')
      return null_value(out);
    return number_value(out);
  }

  bool set_error(const std::string& message) {
    if (error_.empty())
      error_ = message;
    return false;
  }

  bool literal(const char* expected) {
    const size_t length = std::strlen(expected);
    if (text_.compare(pos_, length, expected) != 0)
      return set_error(std::string("expected ") + expected);
    pos_ += length;
    return true;
  }

  bool object_value(Json& out) {
    out.type = Json::Type::Object;
    ++pos_;
    skip_space();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (pos_ < text_.size()) {
      skip_space();
      std::string key;
      if (pos_ >= text_.size() || text_[pos_] != '"')
        return set_error("expected object key");
      if (!string_value(key))
        return false;
      skip_space();
      if (pos_ >= text_.size() || text_[pos_] != ':')
        return set_error("expected ':'");
      ++pos_;
      Json member;
      if (!value(member))
        return false;
      out.object.emplace(std::move(key), std::move(member));
      skip_space();
      if (pos_ >= text_.size())
        return set_error("unterminated object");
      if (text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      if (text_[pos_] != ',')
        return set_error("expected ',' or '}'");
      ++pos_;
    }
    return set_error("unterminated object");
  }

  bool array_value(Json& out) {
    out.type = Json::Type::Array;
    ++pos_;
    skip_space();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (pos_ < text_.size()) {
      Json item;
      if (!value(item))
        return false;
      out.array.push_back(std::move(item));
      skip_space();
      if (pos_ >= text_.size())
        return set_error("unterminated array");
      if (text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      if (text_[pos_] != ',')
        return set_error("expected ',' or ']'");
      ++pos_;
    }
    return set_error("unterminated array");
  }

  static void append_utf8(std::string& out, uint32_t code) {
    if (code <= 0x7f) {
      out.push_back(static_cast<char>(code));
    } else if (code <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
      out.push_back(static_cast<char>(0xe0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
  }

  bool hex4(uint32_t& out) {
    if (pos_ + 4 > text_.size())
      return set_error("short \\u escape");
    out = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      out <<= 4;
      if (c >= '0' && c <= '9')
        out |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        out |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        out |= static_cast<uint32_t>(c - 'A' + 10);
      else
        return set_error("invalid \\u escape");
    }
    return true;
  }

  bool string_value(std::string& out) {
    out.clear();
    if (pos_ >= text_.size() || text_[pos_] != '"')
      return set_error("expected string");
    ++pos_;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"')
        return true;
      if (static_cast<unsigned char>(c) < 0x20)
        return set_error("control character in string");
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size())
        return set_error("unterminated escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t code = 0;
          if (!hex4(code))
            return false;
          append_utf8(out, code);
          break;
        }
        default:
          return set_error("invalid escape");
      }
    }
    return set_error("unterminated string");
  }

  bool bool_value(Json& out) {
    out.type = Json::Type::Bool;
    if (text_[pos_] == 't') {
      if (!literal("true"))
        return false;
      out.boolean = true;
      return true;
    }
    if (!literal("false"))
      return false;
    out.boolean = false;
    return true;
  }

  bool null_value(Json& out) {
    out.type = Json::Type::Null;
    return literal("null");
  }

  bool number_value(Json& out) {
    const size_t begin = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+'))
      ++pos_;
    bool digits = false;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
      digits = true;
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        digits = true;
        ++pos_;
      }
    }
    if (digits && pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+'))
        ++pos_;
      bool exp_digits = false;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        exp_digits = true;
        ++pos_;
      }
      if (!exp_digits)
        return set_error("invalid number exponent");
    }
    if (!digits)
      return set_error("invalid number");
    const std::string token = text_.substr(begin, pos_ - begin);
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(token.c_str(), &end);
    if (errno != 0 || end == nullptr || *end != '\0')
      return set_error("invalid number");
    out.type = Json::Type::Number;
    out.number = parsed;
    return true;
  }

  const std::string& text_;
  size_t pos_ = 0;
  std::string error_;
};

bool parse_json(const std::string& text, Json& out, std::string& error) {
  JsonParser parser(text);
  return parser.parse(out, error);
}

bool read_file(const std::filesystem::path& path, std::string& out,
               std::string& error) {
  std::ifstream stream(path.string().c_str(), std::ios::binary);
  if (!stream) {
    error = "could not open " + path.string();
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    error = "could not read " + path.string();
    return false;
  }
  out = buffer.str();
  return true;
}

std::string format_number(double value) {
  char buffer[64];
  const int written = std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  if (written <= 0)
    return "0";
  return std::string(buffer, static_cast<size_t>(written));
}

void flatten_json(const Json& value, const std::string& prefix,
                  std::map<std::string, std::string>& out) {
  switch (value.type) {
    case Json::Type::Object:
      for (const auto& entry : value.object) {
        const std::string key =
            prefix.empty() ? entry.first : prefix + "." + entry.first;
        flatten_json(entry.second, key, out);
      }
      break;
    case Json::Type::Array:
      for (size_t i = 0; i < value.array.size(); ++i) {
        const std::string key = prefix + "." + std::to_string(i);
        flatten_json(value.array[i], key, out);
      }
      break;
    case Json::Type::String:
      if (!prefix.empty())
        out[prefix] = value.string;
      break;
    case Json::Type::Number:
      if (!prefix.empty())
        out[prefix] = format_number(value.number);
      break;
    case Json::Type::Bool:
      if (!prefix.empty())
        out[prefix] = value.boolean ? "true" : "false";
      break;
    case Json::Type::Null:
      break;
  }
}

const Json* config_path(const Json& config, std::initializer_list<const char*> path) {
  const Json* current = &config;
  for (const char* segment : path) {
    current = current ? current->find(segment) : nullptr;
  }
  return current;
}

bool config_int(const Json& config, std::initializer_list<const char*> path,
                int64_t& out, std::string& error) {
  const Json* value = config_path(config, path);
  if (!value || value->type != Json::Type::Number) {
    error = "missing numeric config value";
    for (const char* segment : path)
      error += std::string(".") + segment;
    return false;
  }
  out = static_cast<int64_t>(value->number);
  return true;
}

bool config_double(const Json& config, std::initializer_list<const char*> path,
                   double& out, std::string& error) {
  const Json* value = config_path(config, path);
  if (!value || value->type != Json::Type::Number) {
    error = "missing numeric config value";
    for (const char* segment : path)
      error += std::string(".") + segment;
    return false;
  }
  out = value->number;
  return true;
}

bool config_bool(const Json& config, std::initializer_list<const char*> path,
                 bool fallback, std::string& error) {
  const Json* value = config_path(config, path);
  if (!value || value->type == Json::Type::Null)
    return fallback;
  if (value->type != Json::Type::Bool) {
    error = "expected boolean config value";
    for (const char* segment : path)
      error += std::string(".") + segment;
    return false;
  }
  return value->boolean;
}

// ---------------------------------------------------------------------------
// Safetensors header inspection. Only the first 8 bytes plus the JSON header
// are read here; tensor payload bytes stay on disk for tei_load_safetensors.
// ---------------------------------------------------------------------------
struct HeaderTensor {
  std::string name;
  std::vector<int64_t> shape;
  int32_t dtype = 2;
  uint64_t offset = 0;
  uint64_t length = 0;
};

struct ShardHeader {
  uint64_t header_bytes = 0;
  uint64_t data_start = 0;
  uint64_t data_length = 0;
  uint64_t file_size = 0;
  std::vector<HeaderTensor> tensors;
};

uint64_t dtype_bytes(const std::string& dtype, bool& ok) {
  ok = true;
  if (dtype == "F32") return 4;
  if (dtype == "F16" || dtype == "BF16") return 2;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  ok = false;
  return 0;
}

int32_t c_api_dtype(const std::string& dtype) {
  if (dtype == "F32") return MISTRAL3_B66_DTYPE_F32;
  if (dtype == "F16") return MISTRAL3_B66_DTYPE_F16;
  if (dtype == "BF16") return MISTRAL3_B66_DTYPE_BF16;
  return -1;
}

bool read_safetensors_header(const std::filesystem::path& path, ShardHeader& out,
                             std::string& error) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) {
    error = "not a regular safetensors file: " + path.string();
    return false;
  }
  out.file_size = static_cast<uint64_t>(info.st_size);
  std::ifstream stream(path.string().c_str(), std::ios::binary);
  if (!stream) {
    error = "could not open " + path.string();
    return false;
  }
  unsigned char length_bytes[8] = {};
  stream.read(reinterpret_cast<char*>(length_bytes), 8);
  if (!stream) {
    error = "short safetensors length prefix: " + path.string();
    return false;
  }
  out.header_bytes = 0;
  for (int i = 7; i >= 0; --i)
    out.header_bytes = (out.header_bytes << 8) | length_bytes[i];
  if (out.header_bytes == 0 || out.header_bytes > 100000000ULL) {
    error = "invalid safetensors header size: " + path.string();
    return false;
  }
  std::string header(out.header_bytes, '\0');
  stream.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!stream) {
    error = "truncated safetensors header: " + path.string();
    return false;
  }
  Json root;
  if (!parse_json(header, root, error) || root.type != Json::Type::Object) {
    error = "invalid safetensors JSON header: " + path.string();
    return false;
  }
  out.data_start = out.header_bytes + 8;
  if (out.data_start > out.file_size) {
    error = "safetensors header extends past end of file: " + path.string();
    return false;
  }
  out.data_length = out.file_size - out.data_start;
  for (const auto& entry : root.object) {
    if (entry.first == "__metadata__")
      continue;
    const Json& info = entry.second;
    const Json* dtype = info.find("dtype");
    const Json* shape = info.find("shape");
    const Json* offsets = info.find("data_offsets");
    if (!dtype || dtype->type != Json::Type::String || !shape ||
        shape->type != Json::Type::Array || !offsets ||
        offsets->type != Json::Type::Array || offsets->array.size() != 2) {
      error = "malformed tensor header for " + entry.first;
      return false;
    }
    HeaderTensor tensor;
    tensor.name = entry.first;
    bool known = false;
    const uint64_t element_bytes = dtype_bytes(dtype->string, known);
    if (!known) {
      error = "unsupported tensor dtype " + dtype->string;
      return false;
    }
    tensor.dtype = c_api_dtype(dtype->string);
    uint64_t expected_bytes = element_bytes;
    tensor.shape.reserve(shape->array.size());
    for (const Json& dim : shape->array) {
      if (dim.type != Json::Type::Number || dim.number < 0) {
        error = "invalid tensor shape for " + entry.first;
        return false;
      }
      const uint64_t value = static_cast<uint64_t>(dim.number);
      tensor.shape.push_back(static_cast<int64_t>(value));
      if (value != 0 && expected_bytes > std::numeric_limits<uint64_t>::max() / value) {
        error = "tensor shape overflow for " + entry.first;
        return false;
      }
      expected_bytes *= value;
    }
    if (offsets->array[0].type != Json::Type::Number ||
        offsets->array[1].type != Json::Type::Number ||
        offsets->array[0].number < 0 ||
        offsets->array[1].number < offsets->array[0].number) {
      error = "invalid tensor offsets for " + entry.first;
      return false;
    }
    tensor.offset = static_cast<uint64_t>(offsets->array[0].number);
    tensor.length = static_cast<uint64_t>(offsets->array[1].number) - tensor.offset;
    if (tensor.length != expected_bytes ||
        tensor.offset > out.data_length ||
        tensor.length > out.data_length - tensor.offset) {
      error = "tensor offset/length mismatch for " + entry.first;
      return false;
    }
    out.tensors.push_back(std::move(tensor));
  }
  std::sort(out.tensors.begin(), out.tensors.end(),
            [](const HeaderTensor& a, const HeaderTensor& b) {
              return a.offset < b.offset;
            });
  uint64_t previous_end = 0;
  for (const HeaderTensor& tensor : out.tensors) {
    if (tensor.offset < previous_end) {
      error = "overlapping tensor ranges in " + path.string();
      return false;
    }
    previous_end = tensor.offset + tensor.length;
  }
  if (previous_end != out.data_length) {
    error = "safetensors payload does not fill the declared file: " + path.string();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------
struct EngineState {
  void* native = nullptr;
  std::string checkpoint;
  std::string device;
  std::filesystem::path config_path;
  Json config;
  int64_t text_hidden = 0;
  int64_t image_token_index = 10;
  int64_t patch_size = 14;
  int64_t merge_size = 2;
  size_t tensor_count = 0;
  size_t shard_count = 0;
  bool loaded = false;
  bool ready = false;
  std::mutex mutex;
  mistral3_b66_perf_counters perf{};
};

EngineState* state_of(mistral3_b66_handle handle) {
  return static_cast<EngineState*>(handle);
}

bool expected_elements(const EngineState& engine,
                       const mistral3_b66_text_request& request,
                       size_t& expected, std::string& error) {
  if (request.batch <= 0 || !request.cumulative) {
    error = "invalid batch or cumulative offsets";
    return false;
  }
  if (request.pool != MISTRAL3_B66_POOL_CLASSIFICATION &&
      request.pool != MISTRAL3_B66_POOL_MEAN &&
      request.pool != MISTRAL3_B66_POOL_FIRST &&
      request.pool != MISTRAL3_B66_POOL_LAST) {
    error = "unsupported pool selector";
    return false;
  }
  const uint64_t width =
      request.pool == MISTRAL3_B66_POOL_CLASSIFICATION
          ? 1ULL
          : static_cast<uint64_t>(engine.text_hidden);
  uint64_t rows = 0;
  for (size_t i = 0; i < request.pooled_count; ++i) {
    if (!request.pooled || request.pooled[i] < 0 || request.pooled[i] >= request.batch) {
      error = "pooled row out of range";
      return false;
    }
    rows += 1;
  }
  for (size_t i = 0; i < request.raw_count; ++i) {
    if (!request.raw || request.raw[i] < 0 || request.raw[i] >= request.batch) {
      error = "raw row out of range";
      return false;
    }
    const int64_t start = request.cumulative[request.raw[i]];
    const int64_t end = request.cumulative[request.raw[i] + 1];
    if (end < start) {
      error = "raw row has negative length";
      return false;
    }
    rows += static_cast<uint64_t>(end - start);
  }
  if (rows != 0 && width > std::numeric_limits<uint64_t>::max() / rows) {
    error = "output element count overflow";
    return false;
  }
  expected = static_cast<size_t>(rows * width);
  return true;
}

// Convert the packet image into the frozen b66 descriptor.
bool convert_image(const mistral3_b66_image& input, TeiImage& output,
                   std::string& error) {
  if (input.grid_thw[0] != 1 || input.grid_thw[1] <= 0 ||
      input.grid_thw[2] <= 0 || input.merge_size <= 0 ||
      input.patch_dim <= 0 || input.rows <= 0 || !input.pixels) {
    error = "invalid image descriptor";
    return false;
  }
  if (input.rows != input.grid_thw[1] * input.grid_thw[2]) {
    error = "image rows do not match the patch grid";
    return false;
  }
  const uint64_t expected_pixels =
      static_cast<uint64_t>(input.rows) * static_cast<uint64_t>(input.patch_dim);
  if (input.pixel_count != 0 && input.pixel_count != expected_pixels) {
    error = "image pixel_count does not match rows * patch_dim";
    return false;
  }
  if (input.token_start < input.sequence_start || input.token_count <= 0) {
    error = "invalid image token span";
    return false;
  }
  output = TeiImage{};
  output.pixels = input.pixels;
  output.rows = input.rows;
  output.patch_dim = input.patch_dim;
  output.grid[0] = input.grid_thw[0];
  output.grid[1] = input.grid_thw[1];
  output.grid[2] = input.grid_thw[2];
  output.merge_size = input.merge_size;
  output.token_start = input.token_start;
  output.token_count = input.token_count;
  output.sequence_start = input.sequence_start;
  std::memcpy(output.feature_key, input.feature_key, sizeof(output.feature_key));
  output.has_feature_key = input.has_feature_key ? 1 : 0;
  return true;
}

bool convert_images(const mistral3_b66_image* images, size_t image_count,
                    std::vector<TeiImage>& output, std::string& error) {
  output.clear();
  if (image_count != 0 && !images) {
    error = "missing image descriptor array";
    return false;
  }
  output.reserve(image_count);
  for (size_t i = 0; i < image_count; ++i) {
    TeiImage image{};
    if (!convert_image(images[i], image, error))
      return false;
    output.push_back(image);
  }
  return true;
}

bool validate_request_buffers(const mistral3_b66_text_request& request,
                              std::string& error) {
  if (request.batch <= 0 || !request.ids || !request.types ||
      !request.positions || !request.cumulative) {
    error = "missing request buffers";
    return false;
  }
  if (request.cumulative[0] != 0) {
    error = "cumulative offsets must start at zero";
    return false;
  }
  for (int64_t row = 0; row < request.batch; ++row) {
    if (request.cumulative[row + 1] < request.cumulative[row]) {
      error = "cumulative offsets are not monotonic";
      return false;
    }
  }
  if (request.max_sequence <= 0 ||
      request.cumulative[request.batch] > request.max_sequence * request.batch) {
    error = "invalid max_sequence";
    return false;
  }
  if (request.pooled_count && !request.pooled) {
    error = "missing pooled rows";
    return false;
  }
  if (request.raw_count && !request.raw) {
    error = "missing raw rows";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reference cases
// ---------------------------------------------------------------------------
struct PreparedCase {
  std::vector<int64_t> ids;
  std::vector<int64_t> types;
  std::vector<int64_t> positions;
  std::vector<int32_t> cumulative;
  std::vector<int64_t> pooled;
  std::vector<int64_t> raw;
  std::vector<float> pixels;
  mistral3_b66_image image{};
  mistral3_b66_text_request request{};
  bool has_image = false;
  size_t expected = 0;

  void bind() {
    request.ids = ids.data();
    request.types = types.data();
    request.positions = positions.data();
    request.cumulative = cumulative.data();
    request.batch = static_cast<int64_t>(cumulative.empty() ? 0 : cumulative.size() - 1);
    request.pooled = pooled.empty() ? nullptr : pooled.data();
    request.pooled_count = pooled.size();
    request.raw = raw.empty() ? nullptr : raw.data();
    request.raw_count = raw.size();
    request.output_capacity = expected;
    if (has_image) {
      image.pixels = pixels.data();
      image.pixel_count = pixels.size();
    }
  }
};

bool make_text_case(const EngineState& engine, int64_t tokens, int32_t pool,
                    PreparedCase& out, std::string& error) {
  if (tokens <= 0) {
    error = "text case requires positive tokens";
    return false;
  }
  const uint64_t seed = 0x4d495354ULL;
  out.ids.resize(static_cast<size_t>(tokens));
  out.types.assign(static_cast<size_t>(tokens), 0);
  out.positions.resize(static_cast<size_t>(tokens));
  out.cumulative = {0, static_cast<int32_t>(tokens)};
  out.pooled = {0};
  out.raw.clear();
  out.pixels.clear();
  out.has_image = false;
  for (int64_t i = 0; i < tokens; ++i) {
    out.ids[static_cast<size_t>(i)] =
        static_cast<int64_t>(1 + (seed + static_cast<uint64_t>(i) * 37) % 1000);
    out.positions[static_cast<size_t>(i)] = i;
  }
  out.request.pool = pool;
  out.request.max_sequence = tokens;
  out.bind();
  if (!expected_elements(engine, out.request, out.expected, error))
    return false;
  out.request.output_capacity = out.expected;
  return true;
}

bool make_text_positions_case(const EngineState& engine,
                              const std::vector<int64_t>& positions,
                              int32_t pool, PreparedCase& out,
                              std::string& error) {
  if (positions.empty()) {
    error = "text position case requires positive tokens";
    return false;
  }
  const uint64_t seed = 0x4d495354ULL;
  const int64_t tokens = static_cast<int64_t>(positions.size());
  out.ids.resize(static_cast<size_t>(tokens));
  out.types.assign(static_cast<size_t>(tokens), 0);
  out.positions.clear();
  out.cumulative = {0, static_cast<int32_t>(tokens)};
  out.pooled = {0};
  out.raw.clear();
  out.pixels.clear();
  out.has_image = false;
  for (int64_t i = 0; i < tokens; ++i) {
    out.ids[static_cast<size_t>(i)] =
        static_cast<int64_t>(1 + (seed + static_cast<uint64_t>(i) * 37) % 1000);
    out.positions.push_back(positions[static_cast<size_t>(i)]);
  }
  out.request.pool = pool;
  out.request.max_sequence = tokens;
  out.bind();
  if (!expected_elements(engine, out.request, out.expected, error))
    return false;
  out.request.output_capacity = out.expected;
  return true;
}

bool make_image_case(const EngineState& engine, int64_t height, int64_t width,
                     int64_t sequence_tokens, int64_t token_start, int32_t pool,
                     PreparedCase& out, std::string& error,
                     int64_t position_offset = 0) {
  if (height <= 0 || width <= 0 || sequence_tokens <= 0 || token_start < 0) {
    error = "invalid image case dimensions";
    return false;
  }
  if (height % engine.merge_size != 0 || width % engine.merge_size != 0) {
    error = "image grid is not divisible by merge size";
    return false;
  }
  const int64_t rows = height * width;
  const int64_t patch_dim = 3 * engine.patch_size * engine.patch_size;
  const int64_t merge_tokens =
      (height / engine.merge_size) * (width / engine.merge_size);
  if (token_start + merge_tokens > sequence_tokens) {
    error = "image token span exceeds the sequence";
    return false;
  }
  out.ids.resize(static_cast<size_t>(sequence_tokens));
  out.types.assign(static_cast<size_t>(sequence_tokens), 0);
  out.positions.resize(static_cast<size_t>(sequence_tokens));
  out.cumulative = {0, static_cast<int32_t>(sequence_tokens)};
  out.pooled = {0};
  out.raw.clear();
  const uint64_t seed = 0x4d495354ULL;
  for (int64_t i = 0; i < sequence_tokens; ++i) {
    out.ids[static_cast<size_t>(i)] =
        static_cast<int64_t>(1 + (seed + static_cast<uint64_t>(i) * 37) % 1000);
    out.positions[static_cast<size_t>(i)] = position_offset + i;
  }
  const int64_t merge_width = width / engine.merge_size;
  for (int64_t i = 0; i < merge_tokens; ++i) {
    const int64_t row = i / merge_width;
    const int64_t column = i % merge_width;
    const int64_t index =
        token_start + row * (merge_width + 1) + column;
    if (index >= 0 && index < sequence_tokens)
      out.ids[static_cast<size_t>(index)] = engine.image_token_index;
  }
  out.pixels.resize(static_cast<size_t>(rows * patch_dim));
  for (size_t i = 0; i < out.pixels.size(); ++i) {
    const uint32_t sample = static_cast<uint32_t>((i * 2654435761u) ^ 0x9e3779b9u);
    out.pixels[i] = static_cast<float>(sample & 0xffffu) / 65535.0f;
  }
  out.image = mistral3_b66_image{};
  out.image.grid_thw[0] = 1;
  out.image.grid_thw[1] = height;
  out.image.grid_thw[2] = width;
  out.image.patch_dim = patch_dim;
  out.image.rows = rows;
  out.image.merge_size = engine.merge_size;
  out.image.token_start = token_start;
  out.image.token_count = merge_tokens;
  out.image.sequence_start = 0;
  out.image.has_feature_key = 0;
  out.has_image = true;
  out.request.pool = pool;
  out.request.max_sequence = sequence_tokens;
  out.bind();
  if (!expected_elements(engine, out.request, out.expected, error))
    return false;
  out.request.output_capacity = out.expected;
  return true;
}

struct CaseResult {
  std::string name;
  size_t elements = 0;
  double checksum = 0;
  uint64_t elapsed = 0;
  bool ok = false;
  std::vector<float> values;
};

bool write_vectors(const std::string& directory, const CaseResult& result,
                   std::string& error) {
  if (directory.empty() || result.values.empty())
    return true;
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec) {
    error = "could not create vector directory: " + ec.message();
    return false;
  }
  const std::filesystem::path path =
      std::filesystem::path(directory) / (result.name + ".f32");
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    error = "could not open vector output " + path.string();
    return false;
  }
  stream.write(reinterpret_cast<const char*>(result.values.data()),
               static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!stream) {
    error = "short write to " + path.string();
    return false;
  }
  return true;
}

bool run_sync_case(EngineState& engine, const PreparedCase& prepared,
                   const std::string& name, CaseResult& result,
                   std::string& error) {
  if (prepared.expected == 0) {
    error = "case has zero expected output";
    return false;
  }
  std::vector<float> output(prepared.expected, 0.0f);
  const auto start = Clock::now();
  const int32_t status =
      mistral3_b66_forward(&engine, &prepared.request,
                           prepared.has_image ? &prepared.image : nullptr,
                           prepared.has_image ? 1 : 0, output.data());
  const uint64_t elapsed = elapsed_ns(start);
  if (status != 0) {
    error = g_last_error.empty() ? "forward failed" : g_last_error;
    return false;
  }
  double checksum = 0;
  for (float value : output)
    checksum += static_cast<double>(value);
  result.name = name;
  result.elements = output.size();
  result.checksum = checksum;
  result.elapsed = elapsed;
  result.ok = true;
  result.values = std::move(output);
  return true;
}

bool run_deferred_case(EngineState& engine, const PreparedCase& prepared,
                       const std::string& name, CaseResult& result,
                       std::string& error) {
  if (prepared.expected == 0) {
    error = "case has zero expected output";
    return false;
  }
  const auto submit_start = Clock::now();
  mistral3_b66_output lease = mistral3_b66_forward_submit(
      &engine, &prepared.request,
      prepared.has_image ? &prepared.image : nullptr, prepared.has_image ? 1 : 0);
  const uint64_t submit_ns = elapsed_ns(submit_start);
  if (!lease) {
    error = g_last_error.empty() ? "forward_submit failed" : g_last_error;
    return false;
  }
  const auto wait_start = Clock::now();
  const float* values = mistral3_b66_output_wait(lease);
  const uint64_t wait_ns = elapsed_ns(wait_start);
  if (!values) {
    error = g_last_error.empty() ? "output_wait failed" : g_last_error;
    mistral3_b66_output_destroy(lease);
    return false;
  }
  size_t elements = 0;
  if (mistral3_b66_output_elements(lease, &elements) != 0 ||
      elements != prepared.expected) {
    error = "deferred output element count mismatch";
    mistral3_b66_output_destroy(lease);
    return false;
  }
  double checksum = 0;
  for (size_t i = 0; i < elements; ++i)
    checksum += static_cast<double>(values[i]);
  result.values.assign(values, values + elements);
  mistral3_b66_output_destroy(lease);
  result.name = name;
  result.elements = elements;
  result.checksum = checksum;
  result.elapsed = submit_ns + wait_ns;
  result.ok = true;
  return true;
}

struct PerfSummary {
  uint64_t min_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_ns = 0;
  uint64_t total_ns = 0;
  uint64_t p50_ns = 0;
  double checksum = 0;
  size_t elements = 0;
  size_t iterations = 0;
};

bool run_perf(EngineState& engine, const PreparedCase& prepared,
              size_t warmup, size_t iterations, bool deferred,
              PerfSummary& summary, std::string& error) {
  if (iterations == 0) {
    error = "perf requires at least one measured iteration";
    return false;
  }
  std::vector<float> output(prepared.expected, 0.0f);
  for (size_t i = 0; i < warmup; ++i) {
    CaseResult warm;
    if (deferred) {
      if (!run_deferred_case(engine, prepared, "warmup", warm, error))
        return false;
    } else if (!run_sync_case(engine, prepared, "warmup", warm, error)) {
      return false;
    }
  }
  std::vector<uint64_t> samples;
  samples.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    if (deferred) {
      const auto start = Clock::now();
      mistral3_b66_output lease = mistral3_b66_forward_submit(
          &engine, &prepared.request,
          prepared.has_image ? &prepared.image : nullptr,
          prepared.has_image ? 1 : 0);
      if (!lease) {
        error = g_last_error.empty() ? "perf submit failed" : g_last_error;
        return false;
      }
      const float* values = mistral3_b66_output_wait(lease);
      if (!values) {
        error = g_last_error.empty() ? "perf wait failed" : g_last_error;
        mistral3_b66_output_destroy(lease);
        return false;
      }
      double checksum = 0;
      for (size_t j = 0; j < prepared.expected; ++j)
        checksum += static_cast<double>(values[j]);
      mistral3_b66_output_destroy(lease);
      const uint64_t ns = elapsed_ns(start);
      samples.push_back(ns);
      summary.checksum = checksum;
    } else {
      const auto start = Clock::now();
      const int32_t status =
          mistral3_b66_forward(&engine, &prepared.request,
                               prepared.has_image ? &prepared.image : nullptr,
                               prepared.has_image ? 1 : 0, output.data());
      const uint64_t ns = elapsed_ns(start);
      if (status != 0) {
        error = g_last_error.empty() ? "perf forward failed" : g_last_error;
        return false;
      }
      samples.push_back(ns);
      double checksum = 0;
      for (float value : output)
        checksum += static_cast<double>(value);
      summary.checksum = checksum;
    }
  }
  std::sort(samples.begin(), samples.end());
  summary.elements = prepared.expected;
  summary.iterations = samples.size();
  summary.min_ns = samples.front();
  summary.max_ns = samples.back();
  for (uint64_t value : samples)
    summary.total_ns += value;
  const size_t middle = samples.size() / 2;
  summary.p50_ns = samples[middle];
  return true;
}

bool build_engine_config(const Json& config, TeiConfig& out, std::string& error) {
  int64_t hidden = 0;
  int64_t heads = 0;
  int64_t layers = 0;
  int64_t intermediate = 0;
  int64_t vocab = 0;
  int64_t positions = 0;
  double epsilon = 0;
  if (!config_int(config, {"text_config", "hidden_size"}, hidden, error) ||
      !config_int(config, {"text_config", "num_attention_heads"}, heads, error) ||
      !config_int(config, {"text_config", "num_hidden_layers"}, layers, error) ||
      !config_int(config, {"text_config", "intermediate_size"}, intermediate, error) ||
      !config_int(config, {"text_config", "vocab_size"}, vocab, error) ||
      !config_int(config, {"text_config", "max_position_embeddings"}, positions, error) ||
      !config_double(config, {"text_config", "rms_norm_eps"}, epsilon, error)) {
    return false;
  }
  out = TeiConfig{};
  out.hidden = hidden;
  out.heads = heads;
  out.layers = layers;
  out.intermediate = intermediate;
  out.vocab = vocab;
  out.positions = positions;
  out.types = 0;
  out.epsilon = epsilon;
  out.activation = 0;
  return true;
}

bool load_all_shards(EngineState& engine, size_t threads, size_t buffer_bytes,
                     int32_t direct_io, std::string& error) {
  const std::filesystem::path index_path =
      std::filesystem::path(engine.checkpoint) / "model.safetensors.index.json";
  std::string index_text;
  if (!read_file(index_path, index_text, error))
    return false;
  Json index;
  if (!parse_json(index_text, index, error))
    return false;
  const Json* weight_map = index.find("weight_map");
  if (!weight_map || weight_map->type != Json::Type::Object) {
    error = "model.safetensors.index.json lacks weight_map";
    return false;
  }
  const Json* metadata = index.find("metadata");
  if (!metadata || !metadata->find("total_size") ||
      metadata->find("total_size")->type != Json::Type::Number) {
    error = "model.safetensors.index.json lacks metadata.total_size";
    return false;
  }
  const uint64_t indexed_total =
      static_cast<uint64_t>(metadata->find("total_size")->number);
  std::set<std::string> shards;
  for (const auto& entry : weight_map->object) {
    if (entry.second.type != Json::Type::String) {
      error = "index shard name is not a string";
      return false;
    }
    if (std::filesystem::path(entry.second.string).filename().string() !=
            entry.second.string ||
        entry.second.string.size() < 13 ||
        entry.second.string.substr(entry.second.string.size() - 13) !=
            ".safetensors") {
      error = "unsafe shard name in index";
      return false;
    }
    shards.insert(entry.second.string);
  }
  uint64_t total_payload = 0;
  size_t loaded_tensors = 0;
  for (const std::string& shard : shards) {
    const std::filesystem::path shard_path =
        std::filesystem::path(engine.checkpoint) / shard;
    ShardHeader header;
    if (!read_safetensors_header(shard_path, header, error))
      return false;
    std::set<std::string> expected_names;
    for (const auto& entry : weight_map->object) {
      if (entry.second.type == Json::Type::String &&
          entry.second.string == shard)
        expected_names.insert(entry.first);
    }
    std::set<std::string> actual_names;
    for (const HeaderTensor& tensor : header.tensors)
      actual_names.insert(tensor.name);
    if (actual_names != expected_names) {
      error = "index/header tensor set mismatch for " + shard;
      return false;
    }
    const int fd = open(shard_path.c_str(), O_RDONLY);
    if (fd < 0) {
      error = "could not open shard " + shard_path.string() + ": " +
              std::strerror(errno);
      return false;
    }
    struct stat info {};
    if (fstat(fd, &info) != 0 ||
        static_cast<uint64_t>(info.st_size) != header.file_size) {
      close(fd);
      error = "shard size changed while reading " + shard_path.string();
      return false;
    }
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> shapes;
    names.reserve(header.tensors.size());
    shapes.reserve(header.tensors.size());
    for (const HeaderTensor& tensor : header.tensors) {
      names.push_back(tensor.name);
      shapes.push_back(tensor.shape);
    }
    std::vector<TeiWeight> weights;
    weights.reserve(header.tensors.size());
    for (size_t i = 0; i < header.tensors.size(); ++i) {
      TeiWeight weight{};
      weight.name = names[i].c_str();
      weight.shape = shapes[i].empty() ? nullptr : shapes[i].data();
      weight.rank = shapes[i].size();
      weight.dtype = header.tensors[i].dtype;
      weight.offset = header.tensors[i].offset;
      weight.length = header.tensors[i].length;
      weights.push_back(weight);
    }
    const int32_t status = tei_load_safetensors(
        engine.native, fd, header.data_start, header.data_length,
        weights.data(), weights.size(), threads, buffer_bytes, direct_io);
    close(fd);
    if (status != 0) {
      capture_native(status);
      error = g_last_error;
      return false;
    }
    loaded_tensors += header.tensors.size();
    total_payload += header.data_length;
    ++engine.shard_count;
  }
  if (loaded_tensors != weight_map->object.size()) {
    error = "loaded tensor count does not match the index";
    return false;
  }
  if (total_payload != indexed_total) {
    error = "loaded payload total does not match metadata.total_size";
    return false;
  }
  engine.tensor_count = loaded_tensors;
  engine.loaded = true;
  return true;
}

double average(uint64_t total, size_t count) {
  return count == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(count);
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                        static_cast<unsigned char>(c));
          out += buffer;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

void print_case_json(const CaseResult& result, bool first) {
  if (!first)
    std::cout << ",\n";
  std::cout << "    {\"name\":\"" << json_escape(result.name)
            << "\",\"ok\":" << (result.ok ? "true" : "false")
            << ",\"elements\":" << result.elements
            << ",\"elapsed_ns\":" << result.elapsed
            << ",\"checksum\":" << format_number(result.checksum) << "}";
}

void print_perf_json(const char* name, const PerfSummary& perf, bool first) {
  if (!first)
    std::cout << ",\n";
  std::cout << "    \"" << name << "\":{"
            << "\"iterations\":" << perf.iterations
            << ",\"min_ns\":" << perf.min_ns
            << ",\"max_ns\":" << perf.max_ns
            << ",\"mean_ns\":" << format_number(average(perf.total_ns, perf.iterations))
            << ",\"p50_ns\":" << perf.p50_ns
            << ",\"elements\":" << perf.elements
            << ",\"checksum\":" << format_number(perf.checksum) << "}";
}

void usage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --checkpoint DIR [--device cuda:0] [--iterations N] [--warmup N]\n";
}

}  // namespace

// ---------------------------------------------------------------------------
// Packet ABI
// ---------------------------------------------------------------------------
extern "C" {

mistral3_b66_handle mistral3_b66_create(const char* checkpoint, const char* device,
                                        int32_t dtype) {
  try {
    g_last_error.clear();
    g_last_error_kind = 0;
    if (!checkpoint || !device || checkpoint[0] == '\0' || device[0] == '\0') {
      fail("checkpoint and device are required");
      return nullptr;
    }
    if (dtype != MISTRAL3_B66_DTYPE_F32 && dtype != MISTRAL3_B66_DTYPE_F16 &&
        dtype != MISTRAL3_B66_DTYPE_BF16) {
      fail("unsupported dtype");
      return nullptr;
    }
    const std::string device_name(device);
    if (device_name.rfind("cuda", 0) == 0) {
      const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
      if (!visible || visible[0] == '\0') {
        fail("refusing CUDA create: CUDA_VISIBLE_DEVICES must be set by Root");
        return nullptr;
      }
    }
    auto state = std::make_unique<EngineState>();
    state->checkpoint = checkpoint;
    state->device = device;
    state->config_path = std::filesystem::path(checkpoint) / "config.json";
    std::string text;
    std::string error;
    if (!read_file(state->config_path, text, error) ||
        !parse_json(text, state->config, error)) {
      fail(error);
      return nullptr;
    }
    int64_t hidden = 0;
    int64_t image_token_index = 10;
    int64_t patch_size = 14;
    int64_t merge_size = 2;
    if (!config_int(state->config, {"text_config", "hidden_size"}, hidden, error) ||
        !config_int(state->config, {"image_token_index"}, image_token_index, error) ||
        !config_int(state->config, {"vision_config", "patch_size"}, patch_size, error) ||
        !config_int(state->config, {"spatial_merge_size"}, merge_size, error)) {
      fail(error);
      return nullptr;
    }
    state->text_hidden = hidden;
    state->image_token_index = image_token_index;
    state->patch_size = patch_size;
    state->merge_size = merge_size;
    TeiConfig config{};
    if (!build_engine_config(state->config, config, error)) {
      fail(error);
      return nullptr;
    }
    const auto start = Clock::now();
    void* native = tei_create(&config, device, dtype);
    state->perf.create_ns = elapsed_ns(start);
    if (!native) {
      g_last_error = tei_error() ? tei_error() : "tei_create failed";
      g_last_error_kind = tei_last_error_kind();
      return nullptr;
    }
    state->native = native;
    return state.release();
  } catch (const std::exception& error) {
    fail(error.what());
    return nullptr;
  } catch (...) {
    fail("unknown C++ exception in create");
    return nullptr;
  }
}

int32_t mistral3_b66_option(mistral3_b66_handle handle, const char* key,
                            const char* value) {
  return guarded([&]() -> int32_t {
    EngineState* state = state_of(handle);
    if (!state || !state->native || !key || !value)
      return fail("invalid option arguments");
    const auto start = Clock::now();
    const int32_t status = tei_option(state->native, key, value);
    state->perf.option_ns += elapsed_ns(start);
    return capture_native(status);
  });
}

int32_t mistral3_b66_load_safetensors(
    mistral3_b66_handle handle, const char* shard_path,
    const mistral3_b66_weight* weights, size_t count, size_t threads,
    size_t buffer_bytes, int32_t direct_io) {
  return guarded([&]() -> int32_t {
    EngineState* state = state_of(handle);
    if (!state || !state->native || !shard_path ||
        (count && !weights)) {
      return fail("invalid load arguments");
    }
    if (threads == 0 || buffer_bytes < 65536)
      return fail("invalid weight-reader settings");
    ShardHeader header;
    std::string error;
    if (!read_safetensors_header(shard_path, header, error))
      return fail(error);
    if (count != header.tensors.size())
      return fail("weight descriptor count does not match the shard header");
    const int fd = open(shard_path, O_RDONLY);
    if (fd < 0)
      return fail("could not open shard: " + std::string(std::strerror(errno)));
    struct stat info {};
    if (fstat(fd, &info) != 0 ||
        static_cast<uint64_t>(info.st_size) != header.file_size) {
      close(fd);
      return fail("shard size changed before load");
    }
    std::vector<TeiWeight> native_weights;
    native_weights.reserve(count);
    uint64_t previous_end = 0;
    for (size_t i = 0; i < count; ++i) {
      if (!weights[i].name || (weights[i].rank && !weights[i].shape)) {
        close(fd);
        return fail("invalid weight descriptor");
      }
      if (weights[i].dtype < 0 || weights[i].dtype > 2 ||
          weights[i].offset < previous_end ||
          weights[i].offset > header.data_length ||
          weights[i].length > header.data_length - weights[i].offset) {
        close(fd);
        return fail("weight descriptor is outside the shard payload");
      }
      previous_end = weights[i].offset + weights[i].length;
      TeiWeight weight{};
      weight.name = weights[i].name;
      weight.shape = weights[i].shape;
      weight.rank = weights[i].rank;
      weight.dtype = weights[i].dtype;
      weight.offset = weights[i].offset;
      weight.length = weights[i].length;
      native_weights.push_back(weight);
    }
    const auto start = Clock::now();
    const int32_t status = tei_load_safetensors(
        state->native, fd, header.data_start, header.data_length,
        native_weights.data(), native_weights.size(), threads, buffer_bytes,
        direct_io);
    close(fd);
    state->perf.load_ns += elapsed_ns(start);
    if (status != 0)
      return capture_native(status);
    state->tensor_count += count;
    ++state->shard_count;
    return 0;
  });
}

int32_t mistral3_b66_ready(mistral3_b66_handle handle) {
  return guarded([&]() -> int32_t {
    EngineState* state = state_of(handle);
    if (!state || !state->native)
      return fail("invalid engine handle");
    const auto start = Clock::now();
    const int32_t status = tei_ready(state->native);
    state->perf.ready_ns += elapsed_ns(start);
    if (status != 0)
      return capture_native(status);
    state->ready = true;
    return 0;
  });
}

int32_t mistral3_b66_forward(mistral3_b66_handle handle,
                             const mistral3_b66_text_request* request,
                             const mistral3_b66_image* images,
                             size_t image_count, float* output) {
  return guarded([&]() -> int32_t {
    EngineState* state = state_of(handle);
    if (!state || !state->native || !state->ready || !request || !output)
      return fail("invalid forward arguments");
    std::string error;
    if (!validate_request_buffers(*request, error))
      return fail(error);
    size_t expected = 0;
    if (!expected_elements(*state, *request, expected, error))
      return fail(error);
    if (request->output_capacity < expected)
      return fail("output capacity is smaller than the requested output");
    std::vector<TeiImage> native_images;
    if (!convert_images(images, image_count, native_images, error))
      return fail(error);
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto start = Clock::now();
    const int32_t status = tei_forward(
        state->native, request->ids, request->types, request->positions,
        request->cumulative, request->batch, request->max_sequence,
        request->pool, request->pooled, request->pooled_count, request->raw,
        request->raw_count, output, request->output_capacity, nullptr,
        native_images.empty() ? nullptr : native_images.data(),
        native_images.size(), nullptr, 0);
    const uint64_t elapsed = elapsed_ns(start);
    state->perf.forward_sync_ns += elapsed;
    ++state->perf.forward_calls;
    state->perf.last_output_elements = expected;
    if (status != 0)
      return capture_native(status);
    double checksum = 0;
    for (size_t i = 0; i < expected; ++i)
      checksum += static_cast<double>(output[i]);
    state->perf.last_output_checksum =
        static_cast<uint64_t>(std::llround(checksum));
    state->perf.min_forward_ns =
        state->perf.min_forward_ns == 0 ? elapsed
                                        : std::min(state->perf.min_forward_ns, elapsed);
    state->perf.max_forward_ns = std::max(state->perf.max_forward_ns, elapsed);
    state->perf.total_forward_ns += elapsed;
    return 0;
  });
}

mistral3_b66_output mistral3_b66_forward_submit(
    mistral3_b66_handle handle, const mistral3_b66_text_request* request,
    const mistral3_b66_image* images, size_t image_count) {
  try {
    g_last_error.clear();
    g_last_error_kind = 0;
    EngineState* state = state_of(handle);
    if (!state || !state->native || !state->ready || !request) {
      fail("invalid submit arguments");
      return nullptr;
    }
    std::string error;
    if (!validate_request_buffers(*request, error)) {
      fail(error);
      return nullptr;
    }
    size_t expected = 0;
    if (!expected_elements(*state, *request, expected, error)) {
      fail(error);
      return nullptr;
    }
    if (request->output_capacity != expected) {
      fail("deferred output capacity must exactly match the requested output");
      return nullptr;
    }
    std::vector<TeiImage> native_images;
    if (!convert_images(images, image_count, native_images, error)) {
      fail(error);
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto start = Clock::now();
    void* lease = tei_forward_submit(
        state->native, request->ids, request->types, request->positions,
        request->cumulative, request->batch, request->max_sequence,
        request->pool, request->pooled, request->pooled_count, request->raw,
        request->raw_count, expected, nullptr,
        native_images.empty() ? nullptr : native_images.data(),
        native_images.size(), nullptr, 0, nullptr);
    const uint64_t elapsed = elapsed_ns(start);
    state->perf.forward_submit_ns += elapsed;
    ++state->perf.deferred_calls;
    if (!lease) {
      capture_native(-1);
      return nullptr;
    }
    auto output = std::make_unique<mistral3_b66_output_lease>();
    output->native = lease;
    output->engine_state = state;
    output->elements = expected;
    return output.release();
  } catch (const std::exception& error) {
    fail(error.what());
    return nullptr;
  } catch (...) {
    fail("unknown C++ exception in forward_submit");
    return nullptr;
  }
}

const float* mistral3_b66_output_wait(mistral3_b66_output output) {
  try {
    if (!output || !output->native) {
      fail("invalid output lease");
      return nullptr;
    }
    const auto start = Clock::now();
    const float* values = tei_output_wait(output->native);
    const uint64_t elapsed = elapsed_ns(start);
    if (output->engine_state)
      static_cast<EngineState*>(output->engine_state)->perf.output_wait_ns += elapsed;
    if (!values) {
      const char* text = tei_error();
      fail(text ? text : "tei_output_wait failed", tei_last_error_kind());
      return nullptr;
    }
    return values;
  } catch (const std::exception& error) {
    fail(error.what());
    return nullptr;
  } catch (...) {
    fail("unknown C++ exception in output_wait");
    return nullptr;
  }
}

int32_t mistral3_b66_output_elements(mistral3_b66_output output,
                                     size_t* elements) {
  return guarded([&]() -> int32_t {
    if (!output || !elements)
      return fail("invalid output lease");
    *elements = output->elements;
    return 0;
  });
}

void mistral3_b66_output_destroy(mistral3_b66_output output) {
  try {
    if (!output)
      return;
    if (output->native) {
      const auto start = Clock::now();
      tei_output_destroy(output->native);
      const uint64_t elapsed = elapsed_ns(start);
      if (output->engine_state)
        static_cast<EngineState*>(output->engine_state)->perf.output_destroy_ns += elapsed;
    }
    delete output;
  } catch (...) {
    fail("exception while destroying output lease");
  }
}

int32_t mistral3_b66_perf(mistral3_b66_handle handle,
                          mistral3_b66_perf_counters* counters) {
  return guarded([&]() -> int32_t {
    EngineState* state = state_of(handle);
    if (!state || !counters)
      return fail("invalid perf arguments");
    *counters = state->perf;
    return 0;
  });
}

int32_t mistral3_b66_reset_perf(mistral3_b66_handle handle) {
  return guarded([&]() -> int32_t {
    EngineState* state = state_of(handle);
    if (!state)
      return fail("invalid engine handle");
    state->perf = mistral3_b66_perf_counters{};
    return 0;
  });
}

const char* mistral3_b66_last_error(void) { return g_last_error.c_str(); }

int32_t mistral3_b66_last_error_kind(void) { return g_last_error_kind; }

void mistral3_b66_destroy(mistral3_b66_handle handle) {
  try {
    EngineState* state = state_of(handle);
    if (!state)
      return;
    if (state->native) {
      const auto start = Clock::now();
      tei_destroy(state->native);
      state->perf.destroy_ns = elapsed_ns(start);
    }
    delete state;
  } catch (...) {
    fail("exception while destroying engine");
  }
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Standalone Root runner. Build with -DMISTRAL3_B66_NO_MAIN for a shared packet.
// ---------------------------------------------------------------------------
#ifndef MISTRAL3_B66_NO_MAIN
int main(int argc, char** argv) {
  const char* checkpoint = nullptr;
  const char* device = "cuda:0";
  const char* emit_vectors = "";
  bool cpu_stage = false;
  size_t iterations = 20;
  size_t warmup = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else if (arg == "--emit-vectors" && i + 1 < argc) {
      emit_vectors = argv[++i];
    } else if (arg == "--cpu-stage") {
      cpu_stage = true;
    } else if (arg == "--iterations" && i + 1 < argc) {
      char* end = nullptr;
      const unsigned long value = std::strtoul(argv[++i], &end, 10);
      if (!end || *end != '\0' || value == 0) {
        std::cerr << "invalid --iterations\n";
        return 2;
      }
      iterations = static_cast<size_t>(value);
    } else if (arg == "--warmup" && i + 1 < argc) {
      char* end = nullptr;
      const unsigned long value = std::strtoul(argv[++i], &end, 10);
      if (!end || *end != '\0') {
        std::cerr << "invalid --warmup\n";
        return 2;
      }
      warmup = static_cast<size_t>(value);
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (!checkpoint) {
    usage(argv[0]);
    return 2;
  }

  const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  if (cpu_stage) {
    if (std::string(device) == "cuda:0")
      device = "cpu";
  } else if (!visible || visible[0] == '\0') {
    std::cerr
        << "refusing to run: CUDA_VISIBLE_DEVICES is empty/unset. Root must "
           "override the CPU-only guard explicitly (for example "
           "CUDA_VISIBLE_DEVICES=0).\n";
    return 2;
  }

  std::cout << "{\n"
            << "  \"source_only_non_claim\": \"not compiled or run during CPU "
               "control; Root builds and runs this later\",\n"
            << "  \"checkpoint\": \"" << json_escape(checkpoint) << "\",\n"
            << "  \"device\": \"" << json_escape(device) << "\",\n"
            << "  \"cuda_visible_devices\": \""
            << json_escape(visible ? visible : "") << "\",\n"
            << "  \"cpu_stage\": " << (cpu_stage ? "true" : "false")
            << ",\n";

  mistral3_b66_handle handle = mistral3_b66_create(checkpoint, device,
                                                   MISTRAL3_B66_DTYPE_BF16);
  if (!handle) {
    std::cerr << "create failed: " << mistral3_b66_last_error() << "\n";
    return 1;
  }

  std::string error;
  {
    std::string text;
    Json config;
    const std::filesystem::path config_path =
        std::filesystem::path(checkpoint) / "config.json";
    if (!read_file(config_path, text, error) ||
        !parse_json(text, config, error)) {
      std::cerr << "config parse failed: " << error << "\n";
      mistral3_b66_destroy(handle);
      return 1;
    }
    std::map<std::string, std::string> options;
    flatten_json(config, "", options);
    for (const auto& entry : options) {
      if (mistral3_b66_option(handle, entry.first.c_str(),
                              entry.second.c_str()) != 0) {
        std::cerr << "option failed for " << entry.first << ": "
                  << mistral3_b66_last_error() << "\n";
        mistral3_b66_destroy(handle);
        return 1;
      }
    }
    // Preserve the qualified Native graph defaults on the actual GPU stage.
    // CPU metadata/quality preflight uses no engine and sets no graph options.
    if (!cpu_stage) {
      const std::pair<const char*, const char*> graph_options[] = {
          {"_cuda_graphs", "true"},
          {"_cuda_graph_max_captures", "8"},
          {"_cuda_graph_entries", "4"},
          {"_cuda_graph_reuse", "3"},
          {"_cuda_graph_max_tokens", "4096"},
          {"_cuda_graph_max_bytes", "536870912"},
      };
      for (const auto& [key, value] : graph_options) {
        if (mistral3_b66_option(handle, key, value) != 0) {
          std::cerr << "graph option failed for " << key << ": "
                    << mistral3_b66_last_error() << "\n";
          mistral3_b66_destroy(handle);
          return 1;
        }
      }
    }
  }

  EngineState* state = state_of(handle);
  if (!load_all_shards(*state, 8, 256ULL << 20, 0, error)) {
    std::cerr << "weight load failed: " << error << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }
  if (mistral3_b66_ready(handle) != 0) {
    std::cerr << "ready failed: " << mistral3_b66_last_error() << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }

  uint64_t graph_ready[7]{};
  const int32_t graph_ready_status =
      tei_graph_stats(handle, graph_ready, 7);
  const int64_t graph_ready_count = tei_graph_count(handle);
  std::cout << "  \"graphs\": true,\n"
            << "  \"graph_policy\": {\"max_captures\":8,\"entries\":4,"
               "\"reuse\":3,\"max_tokens\":4096,"
               "\"max_bytes\":536870912},\n"
            << "  \"graph_count_after_ready\": " << graph_ready_count
            << ",\n"
            << "  \"graph_stats_after_ready\": {\"status\":"
            << graph_ready_status << ",\"hits\":" << graph_ready[0]
            << ",\"misses\":" << graph_ready[1] << ",\"captures\":"
            << graph_ready[2] << ",\"attempts\":" << graph_ready[3]
            << ",\"evictions\":" << graph_ready[4] << ",\"live_bytes\":"
            << graph_ready[5] << ",\"admitted_bytes\":" << graph_ready[6]
            << "},\n"
            << "  \"tensors\": " << state->tensor_count << ",\n"
            << "  \"shards\": " << state->shard_count << ",\n"
            << "  \"reference_cases\": [\n";

  PreparedCase text_mean_16;
  PreparedCase text_mean_128;
  PreparedCase text_score_32;
  PreparedCase image_mean;
  PreparedCase image_score;
  PreparedCase text_yarn_16384;
  PreparedCase text_yarn_32768;
  PreparedCase text_yarn_262140;
  PreparedCase image_yarn_16384;
  CaseResult result;
  bool first_case = true;
  if (!make_text_case(*state, 16, MISTRAL3_B66_POOL_MEAN, text_mean_16, error) ||
      !make_text_case(*state, 128, MISTRAL3_B66_POOL_MEAN, text_mean_128, error) ||
      !make_text_case(*state, 32, MISTRAL3_B66_POOL_CLASSIFICATION,
                      text_score_32, error) ||
      !make_image_case(*state, 4, 6, 12, 2, MISTRAL3_B66_POOL_MEAN,
                       image_mean, error) ||
      !make_image_case(*state, 8, 8, 24, 2, MISTRAL3_B66_POOL_CLASSIFICATION,
                       image_score, error) ||
      !make_text_positions_case(*state, {16383, 16384, 16385, 16386},
                                MISTRAL3_B66_POOL_MEAN, text_yarn_16384, error) ||
      !make_text_positions_case(*state, {32767, 32768, 32769, 32770},
                                MISTRAL3_B66_POOL_MEAN, text_yarn_32768, error) ||
      !make_text_positions_case(*state, {262140, 262141, 262142, 262143},
                                MISTRAL3_B66_POOL_MEAN, text_yarn_262140, error) ||
      !make_image_case(*state, 4, 6, 12, 2, MISTRAL3_B66_POOL_MEAN,
                       image_yarn_16384, error, 16384)) {
    std::cerr << "case construction failed: " << error << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }
  const std::pair<const char*, const PreparedCase*> cases[] = {
      {"text-mean-16", &text_mean_16},
      {"text-mean-128", &text_mean_128},
      {"text-score-32", &text_score_32},
      {"image-mean-1x4x6", &image_mean},
      {"image-score-1x8x8", &image_score},
      {"text-yarn-16384", &text_yarn_16384},
      {"text-yarn-32768", &text_yarn_32768},
      {"text-yarn-262140", &text_yarn_262140},
      {"image-yarn-16384", &image_yarn_16384},
  };
  for (const auto& item : cases) {
    if (!run_sync_case(*state, *item.second, item.first, result, error)) {
      std::cerr << item.first << " failed: " << error << "\n";
      mistral3_b66_destroy(handle);
      return 1;
    }
    print_case_json(result, first_case);
    first_case = false;
    if (!write_vectors(emit_vectors, result, error)) {
      std::cerr << "vector write failed: " << error << "\n";
      mistral3_b66_destroy(handle);
      return 1;
    }
  }
  if (!run_deferred_case(*state, image_mean, "image-mean-1x4x6-deferred",
                         result, error)) {
    std::cerr << "deferred case failed: " << error << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }
  print_case_json(result, first_case);
  if (!write_vectors(emit_vectors, result, error)) {
    std::cerr << "vector write failed: " << error << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }

  PerfSummary sync_perf;
  PerfSummary deferred_perf;
  if (!run_perf(*state, image_mean, warmup, iterations, false, sync_perf,
                error) ||
      !run_perf(*state, image_mean, warmup, iterations, true, deferred_perf,
                error)) {
    std::cerr << "perf loop failed: " << error << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }
  std::cout << "\n  ],\n  \"perf\": {\n";
  print_perf_json("synchronous_image_mean", sync_perf, true);
  print_perf_json("deferred_image_mean", deferred_perf, false);
  std::cout << "\n  },\n";

  uint64_t graph_final[7]{};
  const int32_t graph_final_status = tei_graph_stats(handle, graph_final, 7);
  const int64_t graph_final_count = tei_graph_count(handle);
  std::cout << "  \"graph_count\": " << graph_final_count << ",\n"
            << "  \"graph_stats\": {\"status\":" << graph_final_status
            << ",\"hits\":" << graph_final[0] << ",\"misses\":"
            << graph_final[1] << ",\"captures\":" << graph_final[2]
            << ",\"attempts\":" << graph_final[3] << ",\"evictions\":"
            << graph_final[4] << ",\"live_bytes\":" << graph_final[5]
            << ",\"admitted_bytes\":" << graph_final[6] << "},\n";

  mistral3_b66_perf_counters counters{};
  if (mistral3_b66_perf(handle, &counters) != 0) {
    std::cerr << "perf copy failed: " << mistral3_b66_last_error() << "\n";
    mistral3_b66_destroy(handle);
    return 1;
  }
  std::cout << "  \"perf_counters\": {"
            << "\"create_ns\":" << counters.create_ns
            << ",\"option_ns\":" << counters.option_ns
            << ",\"load_ns\":" << counters.load_ns
            << ",\"ready_ns\":" << counters.ready_ns
            << ",\"forward_sync_ns\":" << counters.forward_sync_ns
            << ",\"forward_submit_ns\":" << counters.forward_submit_ns
            << ",\"output_wait_ns\":" << counters.output_wait_ns
            << ",\"output_destroy_ns\":" << counters.output_destroy_ns
            << ",\"destroy_ns\":" << counters.destroy_ns
            << ",\"forward_calls\":" << counters.forward_calls
            << ",\"deferred_calls\":" << counters.deferred_calls
            << ",\"last_output_elements\":" << counters.last_output_elements
            << ",\"last_output_checksum\":" << counters.last_output_checksum
            << ",\"min_forward_ns\":" << counters.min_forward_ns
            << ",\"max_forward_ns\":" << counters.max_forward_ns
            << ",\"total_forward_ns\":" << counters.total_forward_ns
            << "}\n}\n";
  mistral3_b66_destroy(handle);
  return 0;
}
#endif
