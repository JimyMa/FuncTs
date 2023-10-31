#pragma once

#include <c10/core/ScalarType.h>

#include <unordered_map>

namespace torch {
namespace jit {

/* Scalar types */

extern std::unordered_map<std::string, c10::ScalarType> strToDtype;

}  // namespace jit
}  // namespace torch
