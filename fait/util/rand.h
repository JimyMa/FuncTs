#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

at::Tensor generateRandomTensor(TensorTypePtr type);

}
}  // namespace torch