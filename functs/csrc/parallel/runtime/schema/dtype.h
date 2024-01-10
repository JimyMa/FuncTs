#pragma once

#include <cstddef>
#include <cstdint>

enum BaseType {
  kInt = 0U,
  kUInt = 1U,
  kFloat = 2U,
};

struct DataType {
  explicit DataType(
      BaseType code = kFloat,
      uint8_t bits = 32,
      uint16_t lanes = 1)
      : code(code), bits(bits), lanes(lanes) {}
  BaseType code;
  uint8_t bits;
  uint16_t lanes;
};
