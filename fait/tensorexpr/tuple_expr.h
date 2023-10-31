#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API Tuple : public ExprNode<Tuple> {
 public:
  static ExprHandle make(const std::vector<ExprPtr> elements) {
    return ExprHandle(alloc<Tuple>(elements));
  }
  static ExprHandle make() {
    return ExprHandle(alloc<Tuple>(std::vector<ExprPtr>()));
  }

  // TODO: unique_name
  const std::vector<ExprPtr>& elements() const { return elements_; }

  Tuple(std::vector<ExprPtr> elements)
      : ExprNodeBase(ToDtype(ScalarType::Undefined), kOther),
        elements_(std::move(elements)) {}

 private:
  std::vector<ExprPtr> elements_;
};

template <>
void ExprNode<Tuple, Expr>::accept(IRVisitor* mutator);

template <>
ExprPtr ExprNode<Tuple, Expr>::accept_mutator(IRMutator* mutator);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
