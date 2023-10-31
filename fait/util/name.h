#pragma once

#include <string>

namespace torch {
namespace jit {

class NameGenerator {
 public:
  NameGenerator(const std::string &prefix) : prefix(prefix) {}

  std::string generate() { return prefix + std::to_string(count++); }

 private:
  std::string prefix;
  size_t count = 0;
};

}  // namespace jit
}  // namespace torch