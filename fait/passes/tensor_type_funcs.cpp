#include <ATen/core/jit_type.h>
#include <functional>

#include "refine_types.h"
#include "type_utils.h"

namespace torch {
namespace jit {

#define INFER_PARAMS Node *node, ValueTypeMap &refinedTypes

static OperatorSet creationOps{
    "aten::tensor.float(float t, *, ScalarType? dtype=None, Device? "
    "device=None, bool requires_grad=False) -> Tensor",
    "aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? "
    "memory_format=None) -> Tensor",
};

static c10::Device inferDeviceCreationOps(INFER_PARAMS) {
  // Check if there is source tensor (`self`)
  c10::Device device(c10::kCUDA);
  auto &schema = node->schema();
  auto selfIdx = schema.argumentIndexWithName("self");
  if (selfIdx)
    device = *node->input(*selfIdx)->type()->cast<TensorType>()->device();

  // Check if there is target tensor (`other`)
  auto otherIdx = schema.argumentIndexWithName("other");
  if (otherIdx)
    device = *node->input(*otherIdx)->type()->cast<TensorType>()->device();

  // Check if device is specified as an argument
  auto deviceIdx = schema.argumentIndexWithName("device");
  if (deviceIdx) {
    auto deviceArg = node->input(*deviceIdx);
    auto ival = toIValue(node->input(*deviceIdx));
    if (ival && ival->isDevice())
      device = (*ival).toDevice();
  }

  return device;
}

static OperatorSet convertOrFillOps{
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? "
    "memory_format=None) -> Tensor",
};

static std::unordered_map<TypeKind, c10::ScalarType> typeKindsToScalarTypes{
    {TypeKind::FloatType, c10::kFloat},
    {TypeKind::IntType, c10::kLong},
    {TypeKind::BoolType, c10::kBool},
};

static void updateDtypeFromArgs(Node *node, const FunctionSchema &schema,
                                c10::ScalarType &dtype) {
  //  Check if there is `stop` for `arange`
  auto endIdx = schema.argumentIndexWithName("end");
  if (endIdx)
    dtype = typeKindsToScalarTypes.at(node->input(*endIdx)->type()->kind());

  // Check if there is source tensor (`self`)
  auto selfIdx = schema.argumentIndexWithName("self");
  if (selfIdx)
    dtype = *node->input(*selfIdx)->type()->cast<TensorType>()->scalarType();

  // Check if there is target tensor (`other`)
  auto otherIdx = schema.argumentIndexWithName("other");
  if (otherIdx)
    dtype = *node->input(*otherIdx)->type()->cast<TensorType>()->scalarType();

  // Check if device is specified as an argument
  auto dtypeIdx = schema.argumentIndexWithName("dtype");
  if (dtypeIdx) {
    auto dtypeArg = node->input(*dtypeIdx);
    auto ival = toIValue(node->input(*dtypeIdx));
    if (ival && ival->isInt())
      dtype = c10::ScalarType((*ival).toInt());
  }
}

static c10::ScalarType inferDtypeConvertOrFillOps(INFER_PARAMS) {
  auto dtype = c10::kFloat;
  auto &schema = node->schema();
  updateDtypeFromArgs(node, schema, dtype);
  return dtype;
};

static OperatorSet tensorOps{
    "aten::tensor.float(float t, *, ScalarType? dtype=None, Device? "
    "device=None, bool requires_grad=False) -> Tensor",
    "aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
};

static c10::SymbolicShape inferShapeTensorOps(INFER_PARAMS) {
  auto value = node->input(0);
  auto type = value->type();
  if (value->type()->kind() == TypeKind::ListType) {
    auto len = getListLen(value, refinedTypes);
    if (len) {
      return c10::IntArrayRef({int64_t(*len)});
    } else
      return getRankedShape(1);
  } else {
    return getRankedShape(0);
  }
}

static c10::ScalarType inferDtypeTensorOps(INFER_PARAMS) {
  auto value = node->input(0);
  auto type = value->type();
  auto kind = type->kind();
  if (typeKindsToScalarTypes.count(kind))
    return typeKindsToScalarTypes[kind];
  else if (kind == TypeKind::ListType) {
    auto elemTy = type->cast<ListType>()->getElementType();
    TORCH_CHECK(typeKindsToScalarTypes.count(elemTy->kind()));
    return typeKindsToScalarTypes[elemTy->kind()];
  } else {
    throw typeError("Cannot infer data type for input %", value->debugName(),
                    " of `aten::tensor`", c10::get_backtrace());
  }
}

static OperatorSet fillOps{
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
};

static c10::SymbolicShape inferShapeFillOps(INFER_PARAMS) {
  auto size = getIntList(node->input(0));
  if (!size)
    return {};
  return *size;
}

static OperatorSet bcastOps{
    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::minimum(Tensor self, Tensor other) -> Tensor",
    "aten::maximum(Tensor self, Tensor other) -> Tensor",
    "aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
};

static ShapeDim bcastDim(const ShapeDim &lhs, const ShapeDim &rhs) {
  // Not clear if neither dimension is known
  if (!lhs && !rhs)
    return c10::nullopt;

  // Select the larger dimension size if both are known
  if (lhs && rhs)
    return std::max(*lhs, *rhs);

  // Infer if only one dimension is known
  if (lhs) {
    if (*lhs > 1)
      return lhs;
    else
      return c10::nullopt;
  } else {
    if (*rhs > 1)
      return rhs;
    else
      return c10::nullopt;
  }

  return c10::nullopt;
}

static ShapeVec bcastShape(const ShapeVec &lhs, const ShapeVec &rhs) {
  int64_t lRank = lhs.size(), rRank = rhs.size();
  auto outRank = std::max(lRank, rRank);
  ShapeVec outShape(size_t(outRank), c10::nullopt);
  for (auto i : c10::irange(outRank)) {
    auto lIdx = lRank - 1 - i, rIdx = rRank - 1 - i;
    ShapeDim outDim;
    if (lIdx < 0)
      outDim = rhs.at(rIdx);
    else if (rIdx < 0)
      outDim = lhs.at(lIdx);
    else
      outDim = bcastDim(lhs.at(lIdx), rhs.at(rIdx));
    outShape[outRank - 1 - i] = outDim;
  }
  return std::move(outShape);
}

static c10::SymbolicShape inferShapeBcastOps(INFER_PARAMS) {
  auto lShape = getShape(node->input(0)->type()),
       rShape = getShape(node->input(1)->type());
  if (!lShape || !rShape)
    return {};
  return bcastShape(*lShape, *rShape);
}

static OperatorSet sumOp{
    "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, "
    "ScalarType? dtype=None) -> Tensor",
};

static c10::SymbolicShape inferShapeSumOp(INFER_PARAMS) {
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return {};
  auto dims = getIntList(node->input(1));
  auto keepdim = constant_as<bool>(node->input(2));
  if (!dims || !keepdim)
    return {};
  ShapeVec result;
  for (auto i : c10::irange(selfShape->size())) {
    auto dim = selfShape->at(i);
    if (!std::count(dims->begin(), dims->end(), i))
      result.push_back(dim);
    else if (*keepdim)
      result.push_back(1);
  }
  return result;
}

static OperatorSet linearOp{
  "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
};

static c10::SymbolicShape inferShapeLinearOp(INFER_PARAMS) {
  auto lShape = getShape(node->input(0)->type());
  auto rShape = getShape(node->input(1)->type());
  if (!lShape || !rShape) return {};
  TORCH_CHECK(rShape->size() == 2);
  TORCH_CHECK(rShape->back() == rShape->back());
  std::vector<ShapeDim> outShape(lShape->begin(), lShape->end() - 2);
  outShape.push_back(*(lShape->end() - 2));
  outShape.push_back(*(rShape->end() - 2));
  return outShape;
}

static OperatorSet mmOp{
    "aten::mm(Tensor self, Tensor mat2) -> Tensor",
};

static c10::SymbolicShape inferShapeMmOp(INFER_PARAMS) {
  auto lShape = getShape(node->input(0)->type()),
       rShape = getShape(node->input(1)->type());
  if (!lShape || !rShape)
    return {};
  TORCH_CHECK(lShape->size() == 2);
  TORCH_CHECK(rShape->size() == 2);
  return ShapeVec{lShape->at(0), rShape->at(1)};
}

static OperatorSet matmulOp{
    "aten::matmul(Tensor self, Tensor other) -> Tensor",
};

static c10::SymbolicShape inferShapeMatmulOp(INFER_PARAMS) {
  auto lShape = getShape(node->input(0)->type()),
       rShape = getShape(node->input(1)->type());
  if (!lShape || !rShape)
    return {};
  TORCH_CHECK(lShape->size() >= 2);
  TORCH_CHECK(rShape->size() >= 2);
  std::vector<ShapeDim> lBatchDims(lShape->begin(), lShape->end() - 2),
      rBatchDims(rShape->begin(), rShape->end() - 2);
  auto outShape = bcastShape(lBatchDims, rBatchDims);
  outShape.push_back(*(lShape->end() - 2));
  outShape.push_back(rShape->back());
  return outShape;
}

static OperatorSet selectOp{
    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
    "immut::select(Tensor src, int dim, int index) -> Tensor",
};

static c10::SymbolicShape inferShapeSelectOp(INFER_PARAMS) {
  // Process argument
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal)
    return getRankedShape(rank - 1);
  auto dim = dimIVal->toInt();
  if (dim < 0)
    dim += rank;

  // Infer output shape
  inShape->erase(inShape->begin() + dim);
  return *inShape;
}

static c10::optional<int64_t>
refineDimSizeIndex(Value *indexValue,
                   const c10::optional<int64_t> &defaultIfNone) {
  c10::optional<int64_t> index;
  auto ival = toIValue(indexValue);
  if (!ival)
    return c10::nullopt;
  if (ival->isNone())
    index = defaultIfNone;
  else if (ival->isInt())
    index = ival->toInt();
  return index;
}

static OperatorSet sliceOp{
    "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? "
    "end=None, SymInt step=1) -> Tensor(a)",
    "immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
    "end=None, SymInt step=1) -> Tensor"};

static c10::SymbolicShape inferShapeSliceOp(INFER_PARAMS) {
  // Process argument
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal)
    return getRankedShape(rank);
  auto dim = dimIVal->toInt();
  if (dim < 0)
    dim += rank;

  // Process dimension range
  auto dimSize = inShape->at(dim);
  auto start = refineDimSizeIndex(node->input(2), 0);
  auto end = refineDimSizeIndex(node->input(3), dimSize);
  auto step = refineDimSizeIndex(node->input(4), 1);
  auto outDimSize = tryApply<int64_t>(
      [](int64_t dimSize, int64_t start, int64_t end, int64_t step) {
        if (start < 0)
          start += dimSize;
        if (end < 0)
          end += dimSize;
        return (std::min(end, dimSize) - start - 1) / step + 1;
      },
      dimSize, start, end, step);

  // Compute output shape
  ShapeVec outShape;
  for (auto i : c10::irange(rank)) {
    ShapeDim size;
    if (i == dim)
      size = outDimSize;
    else
      size = inShape->at(i);
    outShape.push_back(size);
  }

  return outShape;
}

static OperatorSet squeezeOp{
    "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
    "immut::squeeze(Tensor self, int dim) -> Tensor"};

static c10::SymbolicShape inferShapeSqueezeOp(INFER_PARAMS) {
  // Process arguments
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal)
    return {};
  auto dim = dimIVal->toInt();
  if (dim < 0)
    dim += rank;

  // Remove dimension from shape
  auto dimSize = inShape->at(dim);
  if (!dimSize)
    return {};
  if (*dimSize == 1)
    inShape->erase(inShape->begin() + dim);
  return *inShape;
}

static OperatorSet unsqueezeOp{
    "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
    "immut::unsqueeze(Tensor self, int dim) -> Tensor"};

static c10::SymbolicShape inferShapeUnsqueezeOp(INFER_PARAMS) {
  // Process arguments
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal)
    return getRankedShape(rank + 1);
  auto dim = dimIVal->toInt();
  if (dim < 0)
    dim += rank + 1;

  // Insert dimension to shape
  inShape->insert(inShape->begin() + dim, 1);
  return *inShape;
}

static OperatorSet reshapeOps{
    "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
    "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)",
    "immut::view(Tensor(a) self, SymInt[] size) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeReshapeOps(INFER_PARAMS) {
  // Get new shape
  auto newShape = getIntList(node->input(1));
  if (!newShape)
    return {};
  if (c10::SymbolicShape(*newShape).isComplete())
    return *newShape;

  // Get self shape
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return *newShape;

  // Try to figure out unknown dimension
  if (!c10::SymbolicShape(*selfShape).isComplete())
    return *newShape;
  size_t numel = 1;
  for (auto &d : *selfShape)
    numel *= *d;
  int64_t unknownDim = numel;
  for (auto &d : *newShape)
    if (*d > 0)
      unknownDim /= *d;
  for (auto &d : *newShape)
    if (*d < 0)
      d = unknownDim;

  return *newShape;
}

static OperatorSet permuteOp{
    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
};

static c10::SymbolicShape inferShapePermuteOp(INFER_PARAMS) {
  // Get self shape and dims
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto dims = getIntList(node->input(1));
  if (!dims)
    return getRankedShape(inShape->size());
  TORCH_CHECK(inShape->size() == dims->size());

  // Permute dimensions
  ShapeVec outShape;
  for (auto i : c10::irange(dims->size())) {
    auto dimIdx = dims->at(i);
    ShapeDim shapeDim;
    if (dimIdx)
      shapeDim = inShape->at(*dimIdx);
    outShape.push_back(shapeDim);
  }

  return outShape;
}

static OperatorSet transposeOp{
    "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeTransposeOps(INFER_PARAMS) {
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return {};
  auto rank = selfShape->size();
  auto dim0 = *constant_as<int64_t>(node->input(1));
  if (dim0 < 0)
    dim0 += rank;
  auto dim1 = *constant_as<int64_t>(node->input(2));
  if (dim1 < 0)
    dim1 += rank;
  auto outShape = *selfShape;
  std::swap(outShape[dim0], outShape[dim1]);
  return outShape;
}

static OperatorSet expandOp{
    "aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> "
    "Tensor(a)",
};

static c10::SymbolicShape inferShapeExpandOp(INFER_PARAMS) {
  // Get shape and expand sizes
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto sizes = getIntList(node->input(1));
  if (!sizes)
    return {};
  auto inRank = int64_t(inShape->size()), sizeLen = int64_t(sizes->size());

  // Compute output shape
  auto outRank = std::max(inRank, sizeLen);
  ShapeVec outShape(outRank, c10::nullopt);
  for (auto i : c10::irange(outRank)) {
    auto inIdx = inRank - 1 - i, sizeIdx = sizeLen - 1 - i,
         outIdx = outRank - 1 - i;
    if (inIdx < 0) {
      outShape[outIdx] = sizes->at(sizeIdx);
      continue;
    }
    if (sizeIdx < 0) {
      outShape[outIdx] = inShape->at(inIdx);
      continue;
    }
    outShape[outIdx] = tryApply<int64_t>(
        [](int64_t inDim, int64_t sizeDim) {
          if (sizeDim < 0)
            return inDim;
          else
            return std::max(inDim, sizeDim);
        },
        inShape->at(inIdx), sizes->at(sizeIdx));
  }

  return outShape;
}

static OperatorSet asOps{
    "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeAsOps(INFER_PARAMS) {
  auto shape = getShape(node->input(1)->type());
  if (shape)
    return *shape;
  else
    return {};
}

static OperatorSet repeatOp{
    "aten::repeat(Tensor self, SymInt[] repeats) -> Tensor"};

static c10::SymbolicShape inferShapeRepeatOp(INFER_PARAMS) {
  // Get shape and repeats
  auto inShape = getShape(node->input(0)->type());
  if (!inShape)
    return {};
  auto repeats = getIntList(node->input(1));
  if (!repeats)
    return {};
  auto inRank = int64_t(inShape->size()), repeatLen = int64_t(repeats->size());

  // Compute output shape
  auto outRank = std::max(inRank, repeatLen);
  ShapeVec outShape(outRank, c10::nullopt);
  for (auto i : c10::irange(outRank)) {
    auto inIdx = inRank - 1 - i, repIdx = repeatLen - 1 - i,
         outIdx = outRank - 1 - i;
    if (inIdx < 0) {
      outShape[outIdx] = repeats->at(repIdx);
      continue;
    }
    if (repIdx < 0) {
      outShape[outIdx] = inShape->at(inIdx);
      continue;
    }
    outShape[outIdx] = tryApply<int64_t>(
        std::multiplies<int64_t>(), inShape->at(inIdx), repeats->at(repIdx));
  }

  return outShape;
}

static OperatorSet combineOps{
    "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::ScalarType inferDtypeCombineOps(INFER_PARAMS) {
  auto listTy = refinedTypes.at(node->input(0));
  auto tensorTy = listTy->containedType(0)->cast<TensorType>();
  return *tensorTy->scalarType();
}

static c10::Device inferDeviceCombineOps(INFER_PARAMS) {
  auto listTy = refinedTypes.at(node->input(0));
  auto tensorTy = listTy->containedType(0)->cast<TensorType>();
  return *tensorTy->device();
}

static OperatorSet catOp{
    "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::SymbolicShape inferShapeCatOp(INFER_PARAMS) {
  // Decide input tensor ranks
  auto listTy = getRefinedType(node->input(0), refinedTypes);
  auto rank = accumAttrFromElements<size_t>(listTy, getRank);
  if (!rank)
    return {};

  // Determine insert dimension
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal)
    return getRankedShape(*rank);
  auto dim = dimIVal->toInt();
  if (dim < 0)
    dim += *rank;

  // Propagate outout shape
  auto defaultShape = c10::VaryingShape<int64_t>(*rank).sizes();
  auto initShape = defaultShape;
  initShape->at(dim) = 0;
  auto shape = *accumAttrFromElements(
      listTy, getShape,
      [&](c10::optional<ShapeVec> &&accum,
          c10::optional<ShapeVec> &&newShape) -> c10::optional<ShapeVec> {
        if (!newShape)
          newShape = defaultShape;
        TORCH_CHECK(accum->size() == newShape->size());
        for (auto i : c10::irange(accum->size())) {
          const auto &accumDim = accum->at(i), &newDim = newShape->at(i);
          ShapeDim outDim;
          if (i == dim)
            outDim = tryApply<int64_t>(std::plus<int64_t>(), accumDim, newDim);
          else
            outDim = joinOpt(accumDim, newDim);
          accum->at(i) = outDim;
        }
        return std::move(accum);
      },
      initShape);

  return shape;
}

static OperatorSet stackOp{
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::SymbolicShape inferShapeStackOp(INFER_PARAMS) {
  // Decide input tensor ranks
  auto listTy = getRefinedType(node->input(0), refinedTypes);
  auto rank = accumAttrFromElements<size_t>(listTy, getRank);
  if (!rank)
    return {};

  // Determine insert dimension
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal)
    return getRankedShape(*rank + 1);
  auto dim = dimIVal->toInt();
  if (dim < 0)
    dim += (*rank + 1);

  // Propagate outout shape
  auto defaultShape = c10::VaryingShape<int64_t>(*rank).sizes();
  auto shape = *accumAttrFromElements(
      listTy, getShape,
      [&](c10::optional<ShapeVec> &&accum,
          c10::optional<ShapeVec> &&newShape) -> c10::optional<ShapeVec> {
        if (!newShape)
          newShape = defaultShape;
        TORCH_CHECK(accum->size() == newShape->size());
        for (auto i : c10::irange(accum->size()))
          accum->at(i) = joinOpt(accum->at(i), newShape->at(i));
        return std::move(accum);
      },
      defaultShape);

  // Insert axis to the group
  auto numTensors = mapOpt<int64_t>(getListLen(node->input(0), refinedTypes),
                                    [](size_t i) { return int64_t(i); });
  shape.insert(shape.begin() + dim, numTensors);

  return shape;
}

static OperatorSet indexOp{
    "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
};

static c10::SymbolicShape inferShapeIndexOp(INFER_PARAMS) {
  // Get tensor shapes
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return {};
  // only support advanced indexing with exactly one tensor in `indices`
  TORCH_CHECK(*getListLen(node->input(1), refinedTypes) == 1);
  auto indexTy = getElementType(getRefinedType(node->input(1), refinedTypes), 0)
                     ->cast<TensorType>();
  if (!indexTy->dim())
    return {};
  auto indexRank = *indexTy->dim();

  // Infer shape according to index data type
  auto indexDtype = indexTy->scalarType();
  TORCH_CHECK(indexDtype.has_value());
  switch (*indexDtype) {
  case c10::kBool: {
    auto result = *selfShape;
    result.erase(result.begin(), result.begin() + indexRank - 1);
    result.at(0) = c10::nullopt;
    return result;
  } break;

  case c10::kLong: {
    auto result = *getShape(indexTy);
    result.insert(result.end(), selfShape->begin() + 1, selfShape->end());
    return result;
  } break;

  default: {
    TORCH_CHECK(false, "Indices data type ", *indexDtype, " not supported");
  }
  }

  return {};
}

static OperatorSet nonzeroOp{
    "aten::nonzero(Tensor self) -> Tensor",
};

static c10::SymbolicShape inferShapeNonzeroOp(INFER_PARAMS) {
  auto inRank = mapOpt<int64_t>(getRank(node->input(0)->type()),
                                [](size_t r) { return int64_t(r); });
  return ShapeVec{c10::nullopt, inRank};
}

static OperatorSet maxPool2dOp{
    "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], "
    "int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
};

static c10::SymbolicShape inferShapeMaxPool2dOp(INFER_PARAMS) {
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return {};
  *(selfShape->end() - 2) = c10::nullopt;
  *(selfShape->end() - 1) = c10::nullopt;
  return *selfShape;
}

inline static int64_t computeConvDimNoStride(int64_t in, int64_t w, int64_t pad,
                                             int64_t dil) {
  return in + 2 * pad - (w - 1) * dil - 1;
}

inline static int64_t computeConvDim(int64_t in, int64_t w, int64_t st,
                                     int64_t pad, int64_t dil) {
  return computeConvDimNoStride(in, w, pad, dil) / st + 1;
}

static OperatorSet conv2dOp{
    "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] "
    "stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> "
    "Tensor",
};

static c10::SymbolicShape inferShapeConv2dOp(INFER_PARAMS) {
  // Process inputs
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return {};
  auto batch = selfShape->at(0), inChan = selfShape->at(1),
       inH = selfShape->at(2), inW = selfShape->at(3);
  auto weightShape = getShape(node->input(1)->type());
  if (!weightShape)
    return getRankedShape(4);
  auto outChan = weightShape->at(0), kH = weightShape->at(2),
       kW = weightShape->at(3);
  auto strides = getIntList(node->input(3)),
       padding = getIntList(node->input(4)),
       dilation = getIntList(node->input(5));
  auto groups = constant_as<int64_t>(node->input(6));
  if (!strides || !padding || !dilation || !groups)
    return getRankedShape(4);

  // Compute spatial dimensions
  auto outH = tryApply<int64_t>(computeConvDim, inH, kH, strides->at(0),
                                padding->at(0), dilation->at(0));
  auto outW = tryApply<int64_t>(computeConvDim, inW, kW, strides->at(1),
                                padding->at(1), dilation->at(1));

  return ShapeVec{batch, outChan, outH, outW};
}

static OperatorSet upsample2dOps{
    "aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool "
    "align_corners, float[]? scale_factors) -> Tensor",
};

static c10::SymbolicShape inferShapeUpsample2dOps(INFER_PARAMS) {
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return {};
  auto outputSizes = getIntList(node->input(1));
  if (!outputSizes)
    return {};
  return ShapeVec{selfShape->at(0), selfShape->at(1), outputSizes->at(0),
                  outputSizes->at(1)};
}

static OperatorSet sameShapeOps{
    "aten::index_put(Tensor self, Tensor[] indices, Tensor values, bool "
    "accumulate=False) -> Tensor",
    "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool "
    "accumulate=False) -> Tensor",
    "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
    "immut::select_rev(Tensor self, Tensor src, int dim, int index) -> Tensor",
    "immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
    "SymInt? end=None, SymInt step=1) -> Tensor",
    "aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? "
    "memory_format=None) -> Tensor",
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=0) -> "
    "Tensor(a)",
    "aten::exp(Tensor self) -> Tensor",
    "aten::log(Tensor self) -> Tensor",
    "aten::tanh(Tensor self) -> Tensor",
    "aten::sin(Tensor self) -> Tensor",
    "aten::cos(Tensor self) -> Tensor",
    "aten::sqrt(Tensor self) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::relu(Tensor self) -> Tensor",
    "aten::tril(Tensor self, int diagonal=0) -> Tensor",
    "aten::triu(Tensor self, int diagonal=0) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
    "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> "
    "Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
    "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor "
    "values, Tensor indices)",
};

static c10::SymbolicShape passSameShape(INFER_PARAMS) {
  return node->input(0)->type()->cast<TensorType>()->symbolic_sizes();
}

static OperatorSet rankZeroOps{
    "aten::max(Tensor self) -> Tensor",
    "aten::min(Tensor self) -> Tensor",
    "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
    "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
    "aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
};

static OperatorSet rankOneOps{
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
    // "torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> "
    // "Tensor",
};

static std::set<std::string> rankOneSymbolString {
  "torchvision::nms"
};

static OperatorSet boolOps{
    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
};

static OperatorSet longOps{
    "aten::nonzero(Tensor self) -> Tensor",
    // "torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> "
    // "Tensor",
};

static std::set<std::string> longOpsSymbolString {
  "torchvision::nms"
};

static OperatorSet reduceOps{
    "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
    "values, Tensor indices)",
    "aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
    "values, Tensor indices)",
    "aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, "
    "*, ScalarType? dtype=None) -> Tensor",
    "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor",
};

static void handleShapeReduce(INFER_PARAMS) {
  // Process inputs
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return;
  auto rank = selfShape->size();
  auto dim = *constant_as<int64_t>(node->input(1));
  if (dim < 0)
    dim += rank;
  auto keepdim = *constant_as<bool>(node->input(2));

  // Compute output shape
  auto outShape = *selfShape;
  if (keepdim)
    outShape[dim] = 1;
  else
    outShape.erase(outShape.begin() + dim);
  setShape(node->output(0), outShape);
  if (node->outputs().size() > 1)
    setShape(node->output(1), outShape);
}

static OperatorSet sortOp{
    "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor "
    "values, Tensor indices)",
};

static void handleShapeSort(INFER_PARAMS) {
  auto shape = getShape(node->input(0)->type());
  if (!shape)
    return;
  setShape(node->output(0), *shape);
  setShape(node->output(1), *shape);
}

static OperatorSet topkOp{
    "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool "
    "sorted=True) -> (Tensor values, Tensor indices)",
};

static void handleShapeTopk(INFER_PARAMS) {
  // Process inputs
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape)
    return;
  auto rank = selfShape->size();
  auto k = constant_as<int64_t>(node->input(1));
  auto dim = *constant_as<int64_t>(node->input(2));
  if (dim < 0)
    dim += rank;

  // Compute output shape
  auto outShape = *selfShape;
  outShape[dim] = k;
  setShape(node->output(0), outShape);
  setShape(node->output(1), outShape);
}

static OperatorSet unique2Op{
    "aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, "
    "bool return_counts=False) -> (Tensor, Tensor, Tensor)",
};

static void handleShapeUnique2Op(INFER_PARAMS) {
  setShape(node->output(0), {-1});
}

static OperatorSet indicesOps{
    "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
    "values, Tensor indices)",
    "aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
    "values, Tensor indices)",
    "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor "
    "values, Tensor indices)",
    "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool "
    "sorted=True) -> (Tensor values, Tensor indices)",
};

static void handleDtypeIndices(INFER_PARAMS) {
  auto dtype = *node->input(0)->type()->cast<TensorType>()->scalarType();
  setDtype(node->output(0), dtype);
  setDtype(node->output(1), c10::kLong);
}

static OperatorSet embeddingOps{
  "aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, "
  "bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
};

static void handleShapeEmbeddingOps(INFER_PARAMS) {
  auto inputShape = getShape(node->input(1)->type());
  auto weightShape = getShape(node->input(0)->type());
  if (!inputShape || !weightShape) return;
  auto embedding_size = weightShape->at(1);
  auto outputShape = *inputShape;
  outputShape.push_back(embedding_size);
  setShape(node->output(0), outputShape);
}

static void handleDtypeEmbeddingOps(INFER_PARAMS) {
  setDtype(node->output(0), *node->input(0)->type()->cast<TensorType>()->scalarType());
}

static std::initializer_list<
    std::pair<OperatorSet, c10::SymbolicShape (*)(INFER_PARAMS)>>
    shapeFuncInit{
        {tensorOps, inferShapeTensorOps},
        {fillOps, inferShapeFillOps},
        {bcastOps, inferShapeBcastOps},
        {sumOp, inferShapeSumOp},
        {mmOp, inferShapeMmOp},
        {linearOp, inferShapeLinearOp},
        {matmulOp, inferShapeMatmulOp},
        {selectOp, inferShapeSelectOp},
        {sliceOp, inferShapeSliceOp},
        {squeezeOp, inferShapeSqueezeOp},
        {unsqueezeOp, inferShapeUnsqueezeOp},
        {reshapeOps, inferShapeReshapeOps},
        {permuteOp, inferShapePermuteOp},
        {transposeOp, inferShapeTransposeOps},
        {expandOp, inferShapeExpandOp},
        {asOps, inferShapeAsOps},
        {repeatOp, inferShapeRepeatOp},
        {catOp, inferShapeCatOp},
        {stackOp, inferShapeStackOp},
        {indexOp, inferShapeIndexOp},
        {nonzeroOp, inferShapeNonzeroOp},
        {maxPool2dOp, inferShapeMaxPool2dOp},
        {conv2dOp, inferShapeConv2dOp},
        {upsample2dOps, inferShapeUpsample2dOps},
        {sameShapeOps, passSameShape},
        {rankZeroOps, [](INFER_PARAMS) { return getRankedShape(0); }},
        {rankOneOps, [](INFER_PARAMS) { return getRankedShape(1); }},
    };

static std::initializer_list<
    std::pair<std::set<std::string>, c10::SymbolicShape (*)(INFER_PARAMS)>>
    shapeFuncInitSymbolString{
        {rankOneSymbolString, [](INFER_PARAMS) { return getRankedShape(1); }},
    };

static std::initializer_list<
    std::pair<OperatorSet, c10::ScalarType (*)(INFER_PARAMS)>>
    dtypeFuncInit{
        {convertOrFillOps, inferDtypeConvertOrFillOps},
        {tensorOps, inferDtypeTensorOps},
        {combineOps, inferDtypeCombineOps},
        {boolOps, [](INFER_PARAMS) { return c10::kBool; }},
        {longOps, [](INFER_PARAMS) { return c10::kLong; }},
    };

static std::initializer_list<
    std::pair<std::set<std::string>, c10::ScalarType (*)(INFER_PARAMS)>>
    dtypeFuncInitSymbolString{
        {longOpsSymbolString, [](INFER_PARAMS) { return c10::kLong; }},
    };

static std::initializer_list<
    std::pair<OperatorSet, c10::Device (*)(INFER_PARAMS)>>
    deviceFuncInit{
        {creationOps, inferDeviceCreationOps},
        {combineOps, inferDeviceCombineOps},
    };

static std::initializer_list<std::pair<OperatorSet, void (*)(INFER_PARAMS)>>
    specialShapeHandlerInit{
        {reduceOps, handleShapeReduce},
        {sortOp, handleShapeSort},
        {topkOp, handleShapeTopk},
        {unique2Op, handleShapeUnique2Op},
        {embeddingOps, handleShapeEmbeddingOps},
    };

static std::initializer_list<std::pair<OperatorSet, void (*)(INFER_PARAMS)>>
    specialDtypeHandlerInit{
        {indicesOps, handleDtypeIndices},
        {embeddingOps, handleDtypeEmbeddingOps},
    };

static bool initialized = false;
OperatorMap<c10::SymbolicShape (*)(INFER_PARAMS)> shapeFuncs;
std::map<std::string, c10::SymbolicShape (*)(INFER_PARAMS)> shapeFuncSymbolString;

OperatorMap<c10::ScalarType (*)(INFER_PARAMS)> dtypeFuncs;
std::map<std::string, c10::ScalarType (*)(INFER_PARAMS)> dtypeFuncSymbolString;

OperatorMap<c10::Device (*)(INFER_PARAMS)> deviceFuncs;
OperatorMap<void (*)(Node *, ValueTypeMap &)> specialShapeHandlers;
OperatorMap<void (*)(Node *, ValueTypeMap &)> specialDtypeHandlers;

void initTensorTypeFuncs() {
  if (initialized)
    return;

  for (auto &pair : shapeFuncInit)
    shapeFuncs.insert(pair.first, pair.second);
  for (auto &pair : shapeFuncInitSymbolString) {
    for (auto &op_string : pair.first) {
      shapeFuncSymbolString.insert({op_string, pair.second});
    }
  }

  for (auto &pair : dtypeFuncInit)
    dtypeFuncs.insert(pair.first, pair.second);

  for (auto &pair : dtypeFuncInitSymbolString) {
    for (auto &op_string : pair.first) {
      dtypeFuncSymbolString.insert({op_string, pair.second});
    }
  }

  for (auto &pair : deviceFuncInit)
    deviceFuncs.insert(pair.first, pair.second);

  for (auto &pair : specialShapeHandlerInit)
    specialShapeHandlers.insert(pair.first, pair.second);
  for (auto &pair : specialDtypeHandlerInit)
    specialDtypeHandlers.insert(pair.first, pair.second);
  initialized = true;
}

} // namespace jit
} // namespace torch
