#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/// @brief Eliminate dead code when TensorSSA operations are present.
/// @param graph The graph to be optimized.
void EliminateDeadCodeTSSA(const std::shared_ptr<Graph> &graph);

/// @brief Eliminate common subexpressions when TensorSSA operations are
/// present.
/// @param graph The graph to be optimized.
void EliminateCommonSubexprTSSA(const std::shared_ptr<Graph> &graph);

/// @brief Perform constant folding of the graph.
/// @param graph The graph to be optimized.
/// @return Whether any new constant is computed.
bool FoldConstantsTSSA(const std::shared_ptr<Graph> &graph);

/// @brief Move computation of loop invariants out of loop body.
/// @param graph The graph to be optimized.
void HoistLoopInvariants(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
