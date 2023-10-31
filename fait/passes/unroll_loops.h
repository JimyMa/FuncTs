#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include "refine_types.h"

namespace torch {
namespace jit {

/// @brief Unroll `Loop`s with carried dependencies.
/// @param graph The graph to be transformed.
void UnrollLoopsWithDeps(const std::shared_ptr<Graph> &graph,
                         ValueTypeMap &refinedTypes);

}  // namespace jit
}  // namespace torch