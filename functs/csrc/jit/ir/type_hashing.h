#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct HashTypeCopy {
  size_t operator()(const TypePtr &type) const;
  size_t operator()(const c10::ConstTypePtr &type) const;
};

struct EqualTypeCopy {
  bool operator()(const TypePtr &a, const TypePtr &b) const;
  bool operator()(const c10::ConstTypePtr &a, const c10::ConstTypePtr &b) const;
};

} // namespace jit
} // namespace torch
