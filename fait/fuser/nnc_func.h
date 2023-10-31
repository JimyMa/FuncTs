//
// Created by jimyma on 2/10/23.
//

#ifndef LONG_TAIL_TSSA_NNC_FUNC_H
#define LONG_TAIL_TSSA_NNC_FUNC_H

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using ShapeVec = std::vector<ExprHandle>;
using ValueExprMap = std::unordered_map<Value *, ExprHandle>;

#define SHAPE_FUNC_PARAMS Node *node, const ValueExprMap &valueToExpr
#define CUSTOM_LOWERING_PARAMS                                           \
  Node *node, const ValueExprMap &valueToExpr, const ShapeVec &outShape, \
      ScalarType outDtype

using NNCShapeFunction = ShapeVec (*)(SHAPE_FUNC_PARAMS);
using CustomLoweringFunction = std::function<Tensor(CUSTOM_LOWERING_PARAMS)>;

extern OperatorSet identicalShapeOps;
extern OperatorMap<NNCShapeFunction> shapeFuncs;

inline ShapeVec computeIdenticalShape(SHAPE_FUNC_PARAMS) {
  return BufHandle(valueToExpr.at(node->input(0)).AsNode<Buf>()).dims();
}

extern OperatorMap<CustomLoweringFunction> customLoweringFuncs;

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch

#endif  // LONG_TAIL_TSSA_NNC_FUNC_H
