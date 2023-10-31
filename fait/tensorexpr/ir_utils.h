#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class ExprHasher {
 public:
  size_t operator()(const ExprPtr &expr) const {
    return provider.hash(expr)._h;
  }

 private:
  mutable HashProvider provider;
};

struct ExprEq {
  bool operator()(const ExprPtr &lhs, const ExprPtr &rhs) const {
    return std::to_string(lhs) == std::to_string(rhs);
  }
};

/// @brief Determine if a loop is reduction loop. A loop is a reduction loop if
/// its induction variable appears in `reduce_args` of any `ReduceOp`.
/// @param loop The for-loop to be checked.
/// @return Whether `loop` is a reduction loop.
bool isReductionLoop(ForPtr loop);

/// @brief Get all buffers that a statement stores.
/// @param stmt The statement to be analyzed.
/// @return A set of stored buffers.
std::unordered_set<BufPtr> getStoredBufs(StmtPtr stmt);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
