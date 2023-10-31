#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/// @brief Count number of memory intensive operators.
/// @param graph The graph to be analyzed.
void CountMemoryIntensiveOps(const std::shared_ptr<Graph> &graph);

/// @brief Count number of `ParallelMap`s and normal loops, and print in stdout.
/// @param graph The graph to be analyzed.
void CountLoops(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch