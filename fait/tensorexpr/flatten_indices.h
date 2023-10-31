#pragma once

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

/// @brief Flatten indices in `Load` and `Store`. This implementation rewrites
/// indices recursively.
/// @param stmt The statemenent to be rewritten.
/// @return Rewrite result.
StmtPtr flattenIndices(StmtPtr stmt);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
