#pragma once
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
void RemoveInplace(std::shared_ptr<Graph> g);
}
} // namespace torch
