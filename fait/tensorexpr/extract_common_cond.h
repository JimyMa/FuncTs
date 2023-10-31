#pragma once

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

/// @brief Extract top k common boolean `CompareSelect` conditions in an
/// expression and wrap them in the outmost k `IfThenElse`.
/// @param expr The expression to be rewritten.
/// @return Rewrite result.
ExprPtr extractCommonCond(ExprPtr expr);

/// @brief Apply `extractCommonCond` to all stop expression in `for` loop.
/// @param stmt The statement to be mutated.
/// @return Mutation result.
StmtPtr refactorForStop(StmtPtr stmt);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
