#pragma once

#include <iostream>

#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/strides.h>
#include <torch/csrc/jit/ir/ir.h>

#include "refine_types.h"
#include "util/common.h"
#include "util/traits.h"

namespace torch {
namespace jit {

using ShapeDim = c10::optional<int64_t>;
using ShapeVec = std::vector<ShapeDim>;

template <class... Args>
inline c10::TypeError typeError(Args&&... args) {
  return error<c10::TypeError>(std::forward<Args>(args)...);
}

inline bool isTensor(Value* v) {
  return v->type()->kind() == TypeKind::TensorType;
}

inline auto getShape(const TypePtr& tensorTy) {
  TORCH_CHECK(tensorTy->kind() == TypeKind::TensorType);
  return tensorTy->cast<TensorType>()->sizes().sizes();
}

inline auto getRank(const TypePtr& tensorTy) {
  TORCH_CHECK(tensorTy->kind() == TypeKind::TensorType);
  return tensorTy->cast<TensorType>()->dim();
}

inline c10::SymbolicShape getRankedShape(size_t rank) {
  return c10::optional<size_t>(rank);
}

inline void setShape(Value* value, const ShapeVec& shape) {
  value->setType(value->type()->cast<TensorType>()->withSymbolicShapes(shape));
}

inline void setDtype(Value* value, c10::ScalarType dtype) {
  value->setType(value->type()->cast<TensorType>()->withScalarType(dtype));
}

inline c10::optional<size_t> getListLen(
    Value* list,
    ValueTypeMap& refinedTypes) {
  auto listTy = getRefinedType(list, refinedTypes);
  if (listTy->kind() == TypeKind::TupleType)
    return listTy->cast<TupleType>()->elements().size();
  else
    return c10::nullopt;
}

inline c10::optional<std::vector<c10::optional<int64_t>>> getIntList(
    Value* value) {
  TORCH_CHECK(*value->type() == *ListType::create(IntType::get()));
  if (isMutated(value))
    return c10::nullopt;
  auto node = value->node();
  switch (node->kind()) {
    case prim::Constant: {
      auto cnstList = toIValue(value)->toIntVector();
      std::vector<c10::optional<int64_t>> retList;
      for (auto c : cnstList)
        retList.emplace_back(c);
      return retList;
    };

    case prim::ListConstruct: {
      std::vector<c10::optional<int64_t>> retList;
      for (auto input : node->inputs()) {
        auto ival = toIValue(input);
        if (ival)
          retList.push_back(ival->toInt());
        else
          retList.push_back(c10::nullopt);
      }
      return retList;
    }

    case aten::size: {
      auto tensor = node->input(0);
      return tensor->type()->cast<TensorType>()->sizes().sizes();
    }
  }
  return c10::nullopt;
}

inline TypePtr getElementType(const TypePtr& type, size_t index) {
  switch (type->kind()) {
    case TypeKind::ListType:
      return type->cast<ListType>()->getElementType();

    case TypeKind::TupleType:
      return type->cast<TupleType>()->elements().at(index);

    default:
      TORCH_CHECK(false, "Unreachable");
  }

  return nullptr;
}

inline TypePtr getUnifiedElementType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ListType:
      return type->cast<ListType>()->getElementType();

    case TypeKind::TupleType: {
      auto elemTy =
          at::unifyTypeList(type->cast<TupleType>()->elements(), std::cout);
      if (!elemTy)
        throw typeError("Cannot unify elements in ", *type);
      return *elemTy;
    }

    default:
      TORCH_CHECK(false, "Unreachable");
  }

  return nullptr;
}

inline TypePtr createRefinedListType(
    const TypePtr& elemType,
    const c10::optional<size_t>& len) {
  if (len)
    return TupleType::create(std::vector<TypePtr>(*len, elemType));
  else
    return ListType::create(elemType);
}

template <
    class AttrType,
    class GetFunc,
    class CombineFunc = decltype(joinOpt<AttrType>)>
inline c10::optional<AttrType> accumAttrFromElements(
    const TypePtr& listTy,
    GetFunc&& getFunc,
    CombineFunc&& combineFunc = joinOpt<AttrType>,
    const c10::optional<AttrType>& initVal = c10::nullopt) {
  switch (listTy->kind()) {
    case TypeKind::ListType:
      return getFunc(listTy->cast<ListType>()->getElementType());

    case TypeKind::TupleType: {
      auto elemTypes = listTy->cast<TupleType>()->elements();
      auto result = initVal;
      for (auto elemTy : elemTypes)
        result = combineFunc(std::move(result), getFunc(elemTy));
      return std::move(result);
    }

    default:
      TORCH_CHECK(false, "Unreachable");
  }
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
