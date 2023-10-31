#include "tuple_expr.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <>
void ExprNode<Tuple, Expr>::accept(IRVisitor* mutator) {}

template <>
ExprPtr ExprNode<Tuple, Expr>::accept_mutator(IRMutator* mutator) {
  return nullptr;
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch