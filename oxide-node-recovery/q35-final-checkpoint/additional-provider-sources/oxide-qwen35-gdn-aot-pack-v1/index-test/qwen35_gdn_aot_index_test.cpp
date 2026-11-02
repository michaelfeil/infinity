
#define INFINITY_GDN_AOT_INDEX_ONLY
#include "/tmp/infinity-oxide-qwen35-gdn-aot-worktree/libs/infinity-v3/backends/oxide/device/native_gdn_aot/qwen35_gdn_aot.cu"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
    ++failures;
  }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

int64_t reference_native_kv(int64_t source, int64_t head, int64_t key, int64_t value,
                            int64_t value_heads, int64_t key_dim, int64_t value_dim) {
  return (((source * value_heads) + head) * key_dim + key) * value_dim + value;
}

int64_t reference_provider_vk(int64_t batch, int64_t head, int64_t value, int64_t key,
                              int64_t value_heads, int64_t key_dim, int64_t value_dim) {
  return (((batch * value_heads) + head) * value_dim + value) * key_dim + key;
}

void test_pack_mapping() {
  const int64_t key_heads = 2;
  const int64_t value_heads = 4;
  const int64_t qk_stride = 256;
  const int64_t conv_stride = (2 * key_heads + value_heads) * 128;
  const int64_t ab_stride = 2 * value_heads;
  for (int64_t token = 0; token < 3; ++token) {
    for (int64_t head = 0; head < key_heads; ++head) {
      for (int64_t column = 0; column < 128; ++column) {
        const int64_t row = token * key_heads + head;
        CHECK(infinity_gdn_aot::packed_q_offset(token, head, column, key_heads) ==
              row * 128 + column);
        CHECK(infinity_gdn_aot::packed_qk_offset(row, 0, column, qk_stride) ==
              row * qk_stride + column);
        CHECK(infinity_gdn_aot::packed_qk_offset(row, 1, column, qk_stride) ==
              row * qk_stride + 128 + column);
      }
    }
    for (int64_t head = 0; head < value_heads; ++head) {
      for (int64_t column = 0; column < 128; ++column) {
        CHECK(infinity_gdn_aot::packed_v_offset(token, head, column, value_heads) ==
              (token * value_heads + head) * 128 + column);
        CHECK(infinity_gdn_aot::packed_conv_value_offset(
                  token, head, column, key_heads, conv_stride) ==
              token * conv_stride + 2 * key_heads * 128 + head * 128 + column);
      }
      const int64_t alpha =
          infinity_gdn_aot::prepared_ab_offset(token, head, 0, value_heads, ab_stride);
      const int64_t beta =
          infinity_gdn_aot::prepared_ab_offset(token, head, 1, value_heads, ab_stride);
      CHECK(alpha == token * ab_stride + head);
      CHECK(beta == token * ab_stride + value_heads + head);
      CHECK(beta - alpha == value_heads);
      if (head > 0)
        CHECK(alpha != (token * value_heads + head) * 2);
    }
  }
}

void test_state_layouts() {
  const int64_t value_heads = 2;
  const int64_t key_dim = 4;
  const int64_t value_dim = 3;
  const int64_t history = 3;

  {
    std::vector<uint32_t> native(history * value_heads * key_dim * value_dim);
    std::vector<uint32_t> provider(native.size());
    for (size_t i = 0; i < native.size(); ++i)
      native[i] = static_cast<uint32_t>(0x10000000u + i);
    for (int64_t b = 0; b < history; ++b)
      for (int64_t h = 0; h < value_heads; ++h)
        for (int64_t k = 0; k < key_dim; ++k)
          for (int64_t v = 0; v < value_dim; ++v)
            provider[infinity_gdn_aot::provider_vk_offset(
                b, h, v, k, value_heads, key_dim, value_dim)] =
                native[infinity_gdn_aot::native_kv_offset(
                    b, h, k, v, value_heads, key_dim, value_dim)];
    for (int64_t b = 0; b < history; ++b)
      for (int64_t h = 0; h < value_heads; ++h)
        for (int64_t v = 0; v < value_dim; ++v)
          for (int64_t k = 0; k < key_dim; ++k)
            CHECK(provider[infinity_gdn_aot::provider_vk_offset(
                      b, h, v, k, value_heads, key_dim, value_dim)] ==
                  native[reference_native_kv(b, h, k, v, value_heads, key_dim,
                                             value_dim)]);
  }

  {
    const int64_t batch = 4;
    const int64_t sources[batch] = {2, 0, 2, 1};
    std::vector<uint32_t> native(history * value_heads * key_dim * value_dim);
    std::vector<uint32_t> provider(batch * value_heads * key_dim * value_dim);
    for (size_t i = 0; i < native.size(); ++i)
      native[i] = static_cast<uint32_t>(0xabc00000u + i);
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t h = 0; h < value_heads; ++h)
        for (int64_t k = 0; k < key_dim; ++k)
          for (int64_t v = 0; v < value_dim; ++v)
            provider[infinity_gdn_aot::provider_vk_offset(
                b, h, v, k, value_heads, key_dim, value_dim)] =
                native[infinity_gdn_aot::native_kv_offset(
                    sources[b], h, k, v, value_heads, key_dim, value_dim)];
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t h = 0; h < value_heads; ++h)
        for (int64_t v = 0; v < value_dim; ++v)
          for (int64_t k = 0; k < key_dim; ++k)
            CHECK(provider[infinity_gdn_aot::provider_vk_offset(
                      b, h, v, k, value_heads, key_dim, value_dim)] ==
                  native[reference_native_kv(sources[b], h, k, v, value_heads,
                                             key_dim, value_dim)]);
    for (int64_t h = 0; h < value_heads; ++h)
      for (int64_t v = 0; v < value_dim; ++v)
        for (int64_t k = 0; k < key_dim; ++k)
          CHECK(provider[infinity_gdn_aot::provider_vk_offset(
                    0, h, v, k, value_heads, key_dim, value_dim)] ==
                provider[infinity_gdn_aot::provider_vk_offset(
                    2, h, v, k, value_heads, key_dim, value_dim)]);
  }

  {
    std::vector<uint32_t> provider(history * value_heads * key_dim * value_dim);
    std::vector<uint32_t> native(provider.size());
    for (size_t i = 0; i < provider.size(); ++i)
      provider[i] = static_cast<uint32_t>(0xfe000000u + i);
    for (int64_t b = 0; b < history; ++b)
      for (int64_t h = 0; h < value_heads; ++h)
        for (int64_t k = 0; k < key_dim; ++k)
          for (int64_t v = 0; v < value_dim; ++v)
            native[infinity_gdn_aot::native_kv_offset(
                b, h, k, v, value_heads, key_dim, value_dim)] =
                provider[infinity_gdn_aot::provider_vk_offset(
                    b, h, v, k, value_heads, key_dim, value_dim)];
    for (int64_t b = 0; b < history; ++b)
      for (int64_t h = 0; h < value_heads; ++h)
        for (int64_t k = 0; k < key_dim; ++k)
          for (int64_t v = 0; v < value_dim; ++v)
            CHECK(native[infinity_gdn_aot::native_kv_offset(
                      b, h, k, v, value_heads, key_dim, value_dim)] ==
                  provider[reference_provider_vk(b, h, v, k, value_heads,
                                                 key_dim, value_dim)]);
  }

  CHECK(infinity_gdn_aot::state_tile_count(0, value_heads, key_dim, value_dim) == 0);
  CHECK(infinity_gdn_aot::state_tile_count(0, 0, 0, 0) == 0);
}

void test_64bit_offsets() {
  const int64_t value_heads = 128;
  const int64_t key_dim = 128;
  const int64_t value_dim = 128;
  const int64_t at_32 = infinity_gdn_aot::native_kv_offset(
      2048, 0, 0, 0, value_heads, key_dim, value_dim);
  CHECK(at_32 == (int64_t(1) << 32));
  CHECK(static_cast<uint32_t>(at_32) == 0u);
  const int64_t below = infinity_gdn_aot::native_kv_offset(
      2047, 0, 0, 0, value_heads, key_dim, value_dim);
  CHECK(below == (int64_t(1) << 32) - (128LL * 128 * 128));
  CHECK(static_cast<uint32_t>(below) == 0xffe00000u);
  CHECK(infinity_gdn_aot::provider_vk_offset(
            2048, 0, 0, 0, value_heads, key_dim, value_dim) == (int64_t(1) << 32));
  const int64_t qk =
      infinity_gdn_aot::packed_qk_offset((int64_t(1) << 24) * 128, 0, 0, 256);
  CHECK(qk == (int64_t(1) << 39));
  CHECK(static_cast<uint32_t>(qk) == 0u);
  const int64_t ab =
      infinity_gdn_aot::prepared_ab_offset(int64_t(1) << 24, 0, 0, 128, 256);
  CHECK(ab == (int64_t(1) << 32));
  const int64_t conv = infinity_gdn_aot::packed_conv_value_offset(
      int64_t(1) << 24, 0, 0, 128, (2 * 128 + 128) * 128);
  CHECK(conv == (int64_t(3) << 38) + 2 * 128 * 128);
  CHECK(conv > static_cast<int64_t>(0xffffffffu));
}

}  // namespace

int main() {
  test_pack_mapping();
  test_state_layouts();
  test_64bit_offsets();
  if (failures != 0) {
    std::fprintf(stderr, "%d index-contract checks failed\n", failures);
    return 1;
  }
  std::printf("index-contract checks passed\n");
  return 0;
}
