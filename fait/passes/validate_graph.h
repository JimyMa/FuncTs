#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/// @brief Check if the graph is well-formed.
/// @param graph The graph to be validated.
void Validate(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
