#pragma once

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

/// @brief Collect all intermediate buffers from a statement.
/// @param stmt The statement to be analyzed.
/// @param outBufs Buffers that are known to be outputs of the statement.
/// @return A vector of unique intermediate buffers.
std::vector<BufPtr> collectIntermBufs(
    StmtPtr stmt, const std::unordered_set<BufPtr> &outBufs);

/// @brief Split the outmost loop at presence of intermediate buffers.
/// @param outerLoop The outmost loop whose variable is task index.
/// @param intermBufs Intermediate buffers.
/// @param resultBufs Result buffers.
/// @return Split results.
std::vector<StmtPtr> splitAtIntermBufs(ForPtr outerLoop,
                                       const std::vector<BufPtr> &intermBufs,
                                       const std::vector<BufPtr> &resultBufs);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch