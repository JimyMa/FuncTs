#pragma once

#include "refine_types.h"

namespace torch {
namespace jit {

/// @brief Canonicalize operations to simplify lowering.
/// @param graph The graph to be processed.
void CanonicalizeOps(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
