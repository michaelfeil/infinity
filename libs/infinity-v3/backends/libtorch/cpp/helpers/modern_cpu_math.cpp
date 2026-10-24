// Adapted from Candle 036f39e CPU operators (Apache-2.0).
// Sources: candle-core/src/{cpu/mod.rs,cpu/avx.rs,op.rs,cpu_backend/mod.rs};
// candle-nn/src/attention/cpu_flash/{varlen.rs,online_softmax.rs}.
// Keep the upstream LICENSE and NOTICE distributed with this library.
// Half rounding and F32 operation order are intentional. Compile this file
// without floating-point contraction; other model translation units are unchanged.
#include "helpers/modern_cpu_math.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace tei::modern_cpu {
namespace {
std::string read(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::string value;
  std::getline(file, value);
  return value;
}

int64_t sharing(const std::string& value) {
  std::istringstream stream(value);
  std::string field;
  int64_t count = 0;
  while (std::getline(stream, field, ',')) {
    auto dash = field.find('-');
    count += dash == std::string::npos
                 ? 1
                 : std::stoll(field.substr(dash + 1)) - std::stoll(field.substr(0, dash)) + 1;
  }
  return count;
}
} // namespace

bool supported_profile() {
  // The exact serial attention path is qualified only with one intra-op thread.
  // Multicore deployments retain ATen until that scope is independently qualified.
  if (at::get_num_threads() != 1)
    return false;
#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  // Instruction availability is only the first requirement. The cache-sharing checks below also
  // select the reference GEMM blocking profile, whose Half rounding depends on K-block boundaries.
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
    return false;
  // gemm-common discovers all sysfs CPUs and divides each cache by its sharing
  // count. Require a homogeneous proved L1 profile rather than a hardcoded core.
  try {
    bool found = false;
    for (const auto& cpu : std::filesystem::directory_iterator("/sys/devices/system/cpu")) {
      auto name = cpu.path().filename().string();
      if (name.size() <= 3 || name.substr(0, 3) != "cpu" || !std::isdigit(name[3]))
        continue;
      auto caches = cpu.path() / "cache";
      if (!std::filesystem::is_directory(caches))
        return false;
      for (const auto& cache : std::filesystem::directory_iterator(caches)) {
        if (read(cache.path() / "level") != "1" || read(cache.path() / "type") != "Data")
          continue;
        auto size = read(cache.path() / "size");
        int64_t bytes = std::stoll(size);
        if (size.ends_with("K"))
          bytes *= 1024;
        else if (size.ends_with("M"))
          bytes *= 1024 * 1024;
        const auto shared = sharing(read(cache.path() / "shared_cpu_list"));
        // Cache size, associativity, and line size scope the reproduced reference blocking recipe.
        // They are not general requirements for ModernBERT CPU inference, which retains
        // ATen outside this profile.
        if (shared <= 0 || bytes / shared != 32768 ||
            read(cache.path() / "ways_of_associativity") != "8" ||
            read(cache.path() / "coherency_line_size") != "64")
          return false;
        found = true;
      }
    }
    return found;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

static at::Tensor scalar_attention(const at::Tensor& q,
                                   const at::Tensor& k,
                                   const at::Tensor& v,
                                   const int32_t* offsets,
                                   int64_t batch,
                                   float scale,
                                   int64_t window) {
  auto qf = q.to(at::kFloat).contiguous(), kf = k.to(at::kFloat).contiguous(),
       vf = v.to(at::kFloat).contiguous();
  auto out = at::empty_like(qf);
  const float *qa = qf.data_ptr<float>(), *ka = kf.data_ptr<float>(), *va = vf.data_ptr<float>();
  float* oa = out.data_ptr<float>();
  const auto heads = q.size(1), dim = q.size(2);
  std::vector<float> acc(dim);
  for (int64_t b = 0; b < batch; b++) {
    auto start = offsets[b], length = offsets[b + 1] - start;
    for (int64_t i = 0; i < length; i++)
      for (int64_t h = 0; h < heads; h++) {
        std::fill(acc.begin(), acc.end(), 0.f);
        float maximum = -INFINITY, sum = 0.f;
        auto qr = qa + ((start + i) * heads + h) * dim;
        auto j0 = window < 0 ? 0 : std::max<int64_t>(0, i - window),
             j1 = window < 0 ? length - 1 : std::min<int64_t>(length - 1, i + window);
        for (int64_t j = j0; j <= j1; j++) {
          auto kr = ka + ((start + j) * heads + h) * dim;
          auto vr = va + ((start + j) * heads + h) * dim;
          float score = 0.f;
          for (int64_t d = 0; d < dim; d++)
            score += qr[d] * kr[d];
          score *= scale;
          if (score > maximum) {
            float rescale = std::exp(maximum - score);
            for (auto& a : acc)
              a *= rescale;
            sum = sum * rescale + 1.f;
            maximum = score;
            for (int64_t d = 0; d < dim; d++)
              acc[d] += vr[d];
          } else {
            float w = std::exp(score - maximum);
            for (int64_t d = 0; d < dim; d++)
              acc[d] += vr[d] * w;
            sum += w;
          }
        }
        float inverse = sum > 0.f ? 1.f / sum : 0.f;
        auto row = oa + ((start + i) * heads + h) * dim;
        for (int64_t d = 0; d < dim; d++)
          row[d] = acc[d] * inverse;
      }
  }
  return out.to(q.scalar_type());
}

// Each SIMD lane is one independent key: preserve sequential dimension
// reduction and scalar online-softmax key order, with no padded token storage.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) static at::Tensor packed_attention(const at::Tensor& q,
                                                                   const at::Tensor& k,
                                                                   const at::Tensor& v,
                                                                   const int32_t* offsets,
                                                                   int64_t batch,
                                                                   float scale,
                                                                   int64_t window) {
  auto qf = q.to(at::kFloat).contiguous(), vf = v.to(at::kFloat).contiguous();
  const auto tokens = q.size(0), heads = q.size(1), dim = q.size(2);
  auto packed_k = at::empty({heads, dim, tokens}, qf.options());
  const auto* source = k.data_ptr<at::Half>();
  auto* ka = packed_k.data_ptr<float>();
  // Direct Half-to-Float repack avoids a second full Float K allocation.
  for (int64_t h = 0; h < heads; ++h)
    for (int64_t d = 0; d < dim; ++d)
      for (int64_t t = 0; t < tokens; ++t)
        ka[(h * dim + d) * tokens + t] =
            float(source[t * k.stride(0) + h * k.stride(1) + d * k.stride(2)]);
  auto out = at::empty_like(qf);
  const auto* qa = qf.data_ptr<float>();
  const auto* va = vf.data_ptr<float>();
  auto* oa = out.data_ptr<float>();
  std::vector<float> acc(dim);
  for (int64_t b = 0; b < batch; ++b) {
    const auto start = offsets[b], length = offsets[b + 1] - start;
    for (int64_t i = 0; i < length; ++i)
      for (int64_t h = 0; h < heads; ++h) {
        std::fill(acc.begin(), acc.end(), 0.f);
        float maximum = -INFINITY, sum = 0.f;
        const auto* qr = qa + ((start + i) * heads + h) * dim;
        const auto j0 = window < 0 ? 0 : std::max<int64_t>(0, i - window);
        const auto j1 = window < 0 ? length - 1 : std::min<int64_t>(length - 1, i + window);
        auto update = [&](int64_t j, float score) {
          const auto* vr = va + ((start + j) * heads + h) * dim;
          if (score > maximum) {
            const float rescale = std::exp(maximum - score);
            for (auto& a : acc)
              a *= rescale;
            sum = sum * rescale + 1.f;
            maximum = score;
            for (int64_t d = 0; d < dim; ++d)
              acc[d] += vr[d];
          } else {
            const float weight = std::exp(score - maximum);
            for (int64_t d = 0; d < dim; ++d)
              acc[d] += vr[d] * weight;
            sum += weight;
          }
        };
        int64_t j = j0;
        for (; j + 7 <= j1; j += 8) {
          auto scores = _mm256_setzero_ps();
          for (int64_t d = 0; d < dim; ++d) {
            const auto keys = _mm256_loadu_ps(ka + (h * dim + d) * tokens + start + j);
            scores = _mm256_add_ps(scores, _mm256_mul_ps(_mm256_set1_ps(qr[d]), keys));
          }
          float values[8];
          _mm256_storeu_ps(values, _mm256_mul_ps(scores, _mm256_set1_ps(scale)));
          for (int64_t lane = 0; lane < 8; ++lane)
            update(j + lane, values[lane]);
        }
        for (; j <= j1; ++j) {
          float score = 0.f;
          for (int64_t d = 0; d < dim; ++d)
            score += qr[d] * ka[(h * dim + d) * tokens + start + j];
          update(j, score * scale);
        }
        const float inverse = sum > 0.f ? 1.f / sum : 0.f;
        auto* row = oa + ((start + i) * heads + h) * dim;
        for (int64_t d = 0; d < dim; ++d)
          row[d] = acc[d] * inverse;
      }
  }
  return out.to(q.scalar_type());
}
#endif
at::Tensor attention(const at::Tensor& q,
                     const at::Tensor& k,
                     const at::Tensor& v,
                     const int32_t* offsets,
                     int64_t batch,
                     float scale,
                     int64_t window) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx2") && q.device().is_cpu() && k.device().is_cpu() &&
      v.device().is_cpu() && q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
      v.scalar_type() == at::kHalf && q.dim() == 3 && k.sizes() == q.sizes() &&
      v.sizes() == q.sizes())
    return packed_attention(q, k, v, offsets, batch, scale, window);
#endif
  return scalar_attention(q, k, v, offsets, batch, scale, window);
}

static float candle_sum(const float* x, int64_t width) {
  float sum[4][8] = {};
  int64_t limit = width & ~31LL;
  for (int64_t i = 0; i < limit; i += 32)
    for (int j = 0; j < 4; j++)
      for (int t = 0; t < 8; t++)
        sum[j][t] += x[i + j * 8 + t];
  float a[8];
  for (int t = 0; t < 8; t++)
    a[t] = (sum[0][t] + sum[1][t]) + (sum[2][t] + sum[3][t]);
  float v[4];
  for (int t = 0; t < 4; t++)
    v[t] = a[t] + a[t + 4];
  float result = (v[0] + v[1]) + (v[2] + v[3]);
  for (int64_t i = limit; i < width; i++)
    result += x[i];
  return result;
}

at::Tensor norm(const at::Tensor& x, const at::Tensor& w, float eps) {
  auto input = x.to(at::kFloat).contiguous(), gain = w.to(at::kFloat).contiguous();
  auto out = at::empty_like(x);
  const auto width = x.size(-1), rows = x.numel() / width;
  auto xp = input.data_ptr<float>(), wp = gain.data_ptr<float>();
  auto op = out.data_ptr<at::Half>();
  std::vector<float> center(width), squared(width);
  float factor = 1.f / width;
  for (int64_t i = 0; i < rows; i++) {
    auto row = xp + i * width;
    float mean = candle_sum(row, width) * factor;
    for (int64_t d = 0; d < width; d++) {
      center[d] = row[d] - mean;
      squared[d] = center[d] * center[d];
    }
    float denom = std::sqrt(candle_sum(squared.data(), width) * factor + eps);
    for (int64_t d = 0; d < width; d++) {
      at::Half normalized(center[d] / denom);
      op[i * width + d] = at::Half(float(normalized) * wp[d]);
    }
  }
  return out;
}

at::Tensor gated_gelu(const at::Tensor& x) {
  auto input = x.contiguous();
  const auto width = x.size(-1) / 2, rows = x.numel() / (2 * width);
  auto out = at::empty({rows, width}, x.options());
  auto xp = input.data_ptr<at::Half>(), op = out.data_ptr<at::Half>();
  at::Half c(.044715f), s(.7978845608028654f);
  for (int64_t i = 0; i < rows; i++)
    for (int64_t d = 0; d < width; d++) {
      auto v = xp[i * 2 * width + d], up = xp[i * 2 * width + width + d];
      at::Half inner(float(c) * float(v));
      inner = at::Half(float(inner) * float(v));
      inner = at::Half(1.f + float(inner));
      at::Half a(float(s) * float(v));
      a = at::Half(float(a) * float(inner));
      a = at::Half(std::tanh(float(a)));
      a = at::Half(1.f + float(a));
      at::Half b(.5f * float(v));
      b = at::Half(float(b) * float(a));
      op[i * width + d] = at::Half(float(b) * float(up));
    }
  return out;
}

at::Tensor linear(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
  // Proved gemm-f16 FMA MR16/NR6, L1 32KiB / 8-way / 64B: auto_kc=512.
  // gemm-common balances K across ceil(K/auto_kc) cache blocks. Each block
  // rounds its accumulated output to Half before the next cache block.
  const auto width = x.size(-1), chunks = (width + 511) / 512, kc = (width + chunks - 1) / chunks;
  at::Tensor output;
  for (int64_t start = 0; start < width; start += kc) {
    auto partial =
        at::matmul(x.slice(-1, start, std::min(width, start + kc)).to(at::kFloat),
                   weight.slice(1, start, std::min(width, start + kc)).to(at::kFloat).t());
    output =
        output.defined() ? (output.to(at::kFloat) + partial).to(at::kHalf) : partial.to(at::kHalf);
  }
  return bias.defined() ? output + bias : output;
}

LinearWeights prepare_linear(const at::Tensor& weight) {
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kHalf && weight.dim() == 2,
              "Modern CPU weight cache requires a loaded CPU Half matrix");
  LinearWeights result{weight.size(1), {}};
  const auto width = result.input_width, chunks = (width + 511) / 512;
  TORCH_CHECK(width > 0, "Modern CPU linear input width must be positive");
  const auto kc = (width + chunks - 1) / chunks;
  for (int64_t start = 0; start < width; start += kc) {
    const auto end = std::min(width, start + kc);
    // Same slice conversion/layout as the uncached path; retain loaded Half rounding.
    result.blocks.push_back({start, end, weight.slice(1, start, end).to(at::kFloat)});
  }
  return result;
}

/// Apply prepared K blocks in reference order, rounding each accumulated partial back to Half. The
/// cache saves weight conversion only; it must not combine blocks into one different GEMM recipe.
at::Tensor linear(const at::Tensor& x, const LinearWeights& weights, const at::Tensor& bias) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kHalf &&
                  x.size(-1) == weights.input_width,
              "Modern CPU cached linear input geometry/dtype mismatch");
  at::Tensor output;
  for (const auto& block : weights.blocks) {
    auto partial = at::matmul(x.slice(-1, block.start, block.end).to(at::kFloat), block.weight.t());
    output =
        output.defined() ? (output.to(at::kFloat) + partial).to(at::kHalf) : partial.to(at::kHalf);
  }
  return bias.defined() ? output + bias : output;
}

std::pair<at::Tensor, at::Tensor> trig(const at::Tensor& phase, at::ScalarType dtype) {
  auto values = phase.contiguous();
  auto input = values.data_ptr<float>();
  std::vector<float> cosine(values.numel()), sine(values.numel());
  for (int64_t i = 0; i < values.numel(); ++i) {
    cosine[i] = std::cos(input[i]);
    sine[i] = std::sin(input[i]);
  }
  return {at::from_blob(cosine.data(), values.sizes(), values.options()).to(dtype),
          at::from_blob(sine.data(), values.sizes(), values.options()).to(dtype)};
}

at::Tensor mean(const at::Tensor& x) {
  auto values = x.contiguous();
  const auto rows = values.size(0), width = values.size(1);
  auto output = at::empty({1, width}, values.options());
  auto input = values.data_ptr<at::Half>();
  auto result = output.data_ptr<at::Half>();
  at::Half factor(1.f / rows);
  for (int64_t d = 0; d < width; ++d) {
    at::Half sum(0.f);
    for (int64_t row = 0; row < rows; ++row)
      sum = at::Half(float(sum) + float(input[row * width + d]));
    result[d] = at::Half(float(sum) * float(factor));
  }
  return output;
}
} // namespace tei::modern_cpu
