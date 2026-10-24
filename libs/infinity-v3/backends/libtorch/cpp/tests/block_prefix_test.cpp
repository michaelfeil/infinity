#include "batching/block_prefix.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  auto require = [](bool ok) {
    if (!ok)
      throw std::runtime_error("Incorrect multimodal block prefix identity");
  };
  std::array<int32_t, 3> offsets{0, 5, 10};
  std::array<int64_t, 10> ids{1, 2, 3, 4, 5, 1, 2, 3, 8, 9}, types{},
      positions{0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
  std::array<int64_t, 30> axes{};
  std::array<float, 24> pixels{}, copy{};
  std::vector<TeiImage> images{{pixels.data(), 4, 6, {1, 2, 2}, 2, 1, 1, 0},
                               {copy.data(), 4, 6, {1, 2, 2}, 2, 6, 1, 5}};
  auto same = [&] {
    return tei::same_block_prefix(0,
                                  1,
                                  3,
                                  ids.data(),
                                  types.data(),
                                  positions.data(),
                                  axes.data(),
                                  10,
                                  offsets.data(),
                                  images.data(),
                                  images.size());
  };
  require(same());
  copy[1] = 1;
  require(!same());
  copy[1] = 0;
  // Identical feature hashes do not authorize sharing changed pixels.
  images[0].has_feature_key = images[1].has_feature_key = 1;
  copy[1] = 2;
  require(!same());
  copy[1] = 0;
  images[1].grid[1] = 4;
  require(!same());
  images[1].grid[1] = 2;
  images[1].token_start = 7;
  require(!same());
  images[1].token_start = 6;
  axes[10 + 6] = 1;
  require(!same());
  axes[10 + 6] = 0;
  types[6] = 1;
  require(!same());
  types[6] = 0;
  positions[6] = 2;
  require(!same());
  positions[6] = 1;
  require(same());
  images.pop_back();
  require(!same());
  images.clear();
  require(same());
  // Suffix-only images may differ without changing a shared text prefix.
  images.push_back({pixels.data(), 4, 6, {1, 2, 2}, 2, 3, 1, 0});
  require(same());
  std::cout
      << "Block prefix identity checks tokens, types, positions, all rotary axes, image pixels/layout/placement and presence\n";
}
