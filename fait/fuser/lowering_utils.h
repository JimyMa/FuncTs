#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>

#include "tensorexpr/tuple_expr.h"
#include "util/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <class T>
inline ExprHandle getScalarExpr(Value* value, const ValueExprMap& valueToExpr) {
  auto cnst = constant_as<T>(value);
  if (cnst)
    return ExprHandle(
        getImmediateByType<T>(c10::CppTypeToScalarType<T>::value, *cnst));
  else
    return valueToExpr.at(value);
}

inline ExprHandle getAnyScalarExpr(Value* value,
                                   const ValueExprMap& valueToExpr) {
  auto ival = toIValue(value);
  if (ival) {
    switch (value->type()->kind()) {
      case TypeKind::IntType:
        return ExprHandle(getImmediateByType(c10::kLong, ival->toInt()));

      case TypeKind::FloatType:
        return ExprHandle(getImmediateByType(c10::kFloat, ival->toDouble()));

      case TypeKind::BoolType:
        return ExprHandle(getImmediateByType(c10::kBool, ival->toBool()));

      default: {
        TORCH_CHECK(false, "Cannot convert type ", *value->type(),
                    " to immediate.");
        return {};
      };
    }
  } else
    return valueToExpr.at(value);
}

template <class T>
inline std::vector<ExprHandle> getScalarExprList(
    Value* value, const ValueExprMap& valueToExpr) {
  TORCH_CHECK(value->type()->kind() == TypeKind::ListType);
  std::vector<ExprHandle> result;
  auto node = value->node();
  if (node->kind() == prim::Constant) {
    auto cnst = toIValue(value);
    for (auto& elem : cnst->toListRef())
      result.emplace_back(getImmediateByType<T>(
          c10::CppTypeToScalarType<T>::value, elem.to<T>()));
  } else if (node->kind() == prim::ListConstruct) {
    for (auto input : node->inputs())
      result.push_back(getScalarExpr<T>(input, valueToExpr));
  } else if (valueToExpr.count(value) &&
             valueToExpr.at(value).AsNode<Tuple>()) {
    auto tuple = valueToExpr.at(value).AsNode<Tuple>();
    for (auto& elem : tuple->elements()) result.emplace_back(elem);
  } else {
    TORCH_CHECK(false, "Cannot get scalar expression list for value ",
                value->debugName());
  }
  return result;
}

inline std::vector<BufHandle> getBufList(Value* value,
                                         const ValueExprMap& valueToExpr) {
  auto node = value->node();
  if (node->kind() != prim::ListConstruct) {
    TORCH_CHECK(false, "Cannot construct buffer list for value defined by ",
                node->kind());
    return {};
  }
  std::vector<BufHandle> bufs;
  for (auto input : node->inputs())
    bufs.emplace_back(valueToExpr.at(input).AsNode<Buf>());
  return bufs;
}

#define GET_BUF_AT(idx) \
  BufHandle(valueToExpr.at(node->input(idx)).AsNode<Buf>())
#define GET_BUF_LIST_AT(idx) getBufList(node->input(idx), valueToExpr)

#define GET_CONST_AT(idx, type) *constant_as<type>(node->input(idx))
#define GET_INT_CONST_AT(idx) GET_CONST_AT(idx, int64_t)
#define GET_BOOL_CONST_AT(idx) GET_CONST_AT(idx, bool)

#define GET_SCALAR_EXPR_AT(idx, type) \
  getScalarExpr<type>(node->input(idx), valueToExpr)
#define GET_INT_EXPR_AT(idx) GET_SCALAR_EXPR_AT(idx, int64_t)
#define GET_ANY_SCALAR_EXPR_AT(idx) \
  getAnyScalarExpr(node->input(idx), valueToExpr)

#define GET_SCALAR_EXPR_LIST_AT(idx, type) \
  getScalarExprList<type>(node->input(idx), valueToExpr)
#define GET_INT_EXPR_LIST_AT(idx) GET_SCALAR_EXPR_LIST_AT(idx, int64_t)

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch