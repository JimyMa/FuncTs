#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
void DumbRemoveInterPrecedureMutation(std::shared_ptr<Graph> graph);
void ConvertToTensorSSA(std::shared_ptr<Graph> graph);
} // namespace jit
} // namespace torch
