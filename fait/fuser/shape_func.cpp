#include "fuser/nnc_func.h"
#include "lowering_utils.h"
#include "tssa_set_ops.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static ShapeVec getScalarShape(SHAPE_FUNC_PARAMS) { return {}; }

static ShapeVec computeAsShape(SHAPE_FUNC_PARAMS) {
  return GET_BUF_AT(1).dims();
}

static ShapeVec computeFillShape(SHAPE_FUNC_PARAMS) {
  return GET_INT_EXPR_LIST_AT(0);
}

static ShapeVec computeArangeShape(SHAPE_FUNC_PARAMS) {
  auto start = GET_INT_EXPR_AT(0), end = GET_INT_EXPR_AT(1);
  return {end - start};
}

static ShapeVec computeBcastShape(SHAPE_FUNC_PARAMS) {
  auto lShape = GET_BUF_AT(0).dims(), rShape = GET_BUF_AT(1).dims();
  int64_t lRank = lShape.size(), rRank = rShape.size();
  auto outRank = std::max(lRank, rRank);
  ShapeVec outShape(outRank, int64_t(0));
  for (auto i : c10::irange(outRank)) {
    auto lIdx = lRank - 1 - i, rIdx = rRank - 1 - i;
    ExprHandle outDim;
    if (lIdx < 0)
      outDim = rShape.at(rIdx);
    else if (rIdx < 0)
      outDim = lShape.at(lIdx);
    else {
      outDim = Max::make(lShape.at(lIdx), rShape.at(rIdx), true);
      outDim =
          IfThenElse::make(lShape.at(lIdx) == int64_t(0), int64_t(0), outDim);
      outDim =
          IfThenElse::make(rShape.at(rIdx) == int64_t(0), int64_t(0), outDim);
    }

    outShape[outRank - 1 - i] = outDim;
  }
  return outShape;
}

static ShapeVec computeReduceDimShape(SHAPE_FUNC_PARAMS) {
  auto selfShape = GET_BUF_AT(0).dims();
  auto rank = selfShape.size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += rank;
  auto keepdim = GET_BOOL_CONST_AT(2);
  auto result = selfShape;
  if (keepdim)
    result.at(dim) = int64_t(1);
  else
    result.erase(result.begin() + dim);
  return result;
}

static ShapeVec computeSelectShape(SHAPE_FUNC_PARAMS) {
  auto src = GET_BUF_AT(0);
  auto rank = src.dims().size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += rank;
  auto result = src.dims();
  result.erase(result.begin() + dim);
  return result;
}

static ShapeVec computeSliceShape(SHAPE_FUNC_PARAMS) {
  // Tensor
  auto src = GET_BUF_AT(0);
  auto rank = src.dims().size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += rank;
  auto dimSize = src.dims().at(dim);

  // Start
  auto startVal = node->input(2);
  ExprHandle start;
  if (startVal->type()->kind() == TypeKind::NoneType)
    start = LongImm::make(0);
  else
    start = getScalarExpr<int64_t>(startVal, valueToExpr);
  start = IfThenElse::make(start >= int64_t(0), Min::make(start, dimSize, true),
                           start + dimSize);

  // End
  auto endVal = node->input(3);
  ExprHandle end;
  if (endVal->type()->kind() == TypeKind::NoneType)
    end = dimSize;
  else
    end = getScalarExpr<int64_t>(endVal, valueToExpr);
  end = IfThenElse::make(end >= int64_t(0), Min::make(end, dimSize, true),
                         end + dimSize);

  // Step
  int64_t step = GET_INT_CONST_AT(4);

  // Shape
  auto result = src.dims();
  result[dim] = (end - start + step - int64_t(1)) / step;

  return result;
}

static ShapeVec computeUnsqueezeShape(SHAPE_FUNC_PARAMS) {
  auto self = GET_BUF_AT(0);
  auto shape = self.dims();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += shape.size() + 1;
  auto outShape = self.dims();
  outShape.insert(outShape.begin() + dim, int64_t(1));
  return outShape;
}

static ShapeVec computeTransposeShape(SHAPE_FUNC_PARAMS) {
  auto selfShape = GET_BUF_AT(0).dims();
  auto rank = selfShape.size();
  auto dim0 = *constant_as<int64_t>(node->input(1)),
       dim1 = *constant_as<int64_t>(node->input(2));
  if (dim0 < 0)
    dim0 += rank;
  if (dim1 < 0)
    dim1 += rank;
  std::swap(selfShape.at(dim0), selfShape.at(dim1));
  return selfShape;
}

static ShapeVec computePermuteShape(SHAPE_FUNC_PARAMS) {
  auto src = GET_BUF_AT(0);
  auto new_index = *constant_as<IntList>(node->input(1));
  auto src_dims = src.dims();
  std::vector<ExprHandle> result;
  for (auto idx : new_index) {
    result.push_back(src_dims[idx]);
  }
  return result;
}

static ShapeVec computeReshapeShape(SHAPE_FUNC_PARAMS) {
  // Count elements in source tensor
  auto src = GET_BUF_AT(0);
  auto srcShape = src.dims();
  auto srcCount =
      std::accumulate(srcShape.begin(), srcShape.end(), LongImm::make(1),
                      std::mem_fn(&ExprHandle::operator*));

  // Count elements in new tensor
  auto result = GET_INT_EXPR_LIST_AT(1);
  auto resultCount = LongImm::make(1);
  for (auto i : c10::irange(result.size())) {
    auto dim = result[i];
    auto imm = dim.AsNode<LongImm>();
    if (!imm || imm->value() != -1)
      resultCount = resultCount * dim;
  }

  // Fix negative dimension
  for (auto i : c10::irange(result.size())) {
    auto imm = result[i].AsNode<LongImm>();
    if (imm && imm->value() == -1) {
      result[i] = srcCount / resultCount;
      break;
    }
  }

  return result;
}

static ShapeVec computeExpandShape(SHAPE_FUNC_PARAMS) {
  // Get input shape and new shape
  auto inShape = GET_BUF_AT(0).dims();
  auto size = GET_INT_EXPR_LIST_AT(1);
  int64_t inRank = inShape.size(), sizeLen = size.size();

  // Get new shape
  auto outRank = std::max(inRank, sizeLen);
  ShapeVec outShape(outRank);
  for (auto i : c10::irange(outRank)) {
    auto inIdx = inRank - 1 - i, sizeIdx = sizeLen - 1 - i,
         outIdx = outRank - 1 - i;
    ExprHandle outDim;
    if (inIdx < 0)
      outDim = size[sizeIdx];
    else if (sizeIdx < 0)
      outDim = inShape[inIdx];
    else
      outDim = IfThenElse::make(size[sizeIdx] < int64_t(0), inShape[inIdx],
                                size[sizeIdx]);
    outShape[outIdx] = outDim;
  }

  return outShape;
}

static ShapeVec computeRepeatShape(SHAPE_FUNC_PARAMS) {
  // Get input shape and repeats
  auto self = GET_BUF_AT(0);
  auto inShape = self.dims();
  int64_t inRank = inShape.size();
  auto repeats = GET_INT_EXPR_LIST_AT(1);
  int64_t repeatLen = repeats.size();

  // Multiply shape dimensions by repeats
  auto outRank = std::max(inRank, repeatLen);
  ShapeVec outShape(outRank);
  for (auto i : c10::irange(outRank)) {
    auto inIdx = inRank - 1 - i, repIdx = repeatLen - 1 - i,
         outIdx = outRank - 1 - i;
    ExprHandle outDim;
    if (inIdx < 0)
      outDim = repeats[repIdx];
    else if (repIdx < 0)
      outDim = inShape[inIdx];
    else
      outDim = inShape[inIdx] * repeats[repIdx];
    outShape[outIdx] = outDim;
  }

  return outShape;
}

static ShapeVec computeIndexShape(SHAPE_FUNC_PARAMS) {
  auto self = GET_BUF_AT(0);
  auto selfShape = self.dims();
  auto index = GET_BUF_LIST_AT(1).front();
  TORCH_CHECK(index.dtype().scalar_type() == c10::kLong);
  auto result = index.dims();
  result.insert(result.end(), selfShape.begin() + 1, selfShape.end());
  return result;
}

static ShapeVec computeCatShape(SHAPE_FUNC_PARAMS) {
  auto tensors = GET_BUF_LIST_AT(0);
  auto shape = tensors.front().dims();
  auto inRank = shape.size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += inRank;
  auto catDim = shape[dim];
  for (auto i : c10::irange(1, tensors.size()))
    catDim = catDim + tensors[i].dim(dim);
  shape[dim] = catDim;
  return shape;
}

static ShapeVec computeStackShape(SHAPE_FUNC_PARAMS) {
  auto tensors = GET_BUF_LIST_AT(0);
  auto inShape = tensors.front().dims();
  auto inRank = inShape.size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += inRank + 1;
  auto outShape = inShape;
  outShape.insert(outShape.begin() + dim, int64_t(tensors.size()));
  return outShape;
}

static auto _tssaSetOps = registerTssaSetOps();

OperatorMap<NNCShapeFunction> shapeFuncs{
    {"aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
     "bool requires_grad=False) -> Tensor",
     getScalarShape},
    {"aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
     "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
     computeFillShape},
    {"aten::arange.start(Scalar start, Scalar end, *, ScalarType? "
     "dtype=None, Layout? layout=None, Device? device=None, bool? "
     "pin_memory=None) -> Tensor",
     computeArangeShape},
    {"aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
     computeBcastShape},
    {"aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
     computeBcastShape},
    {"aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
     computeBcastShape},
    {"aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
     computeBcastShape},
    {"aten::maximum(Tensor self, Tensor other) -> Tensor", computeBcastShape},
    {"aten::minimum(Tensor self, Tensor other) -> Tensor", computeBcastShape},
    {"aten::eq.Tensor(Tensor self, Tensor other) -> Tensor", computeBcastShape},
    {"aten::lt.Tensor(Tensor self, Tensor other) -> Tensor", computeBcastShape},
    {"aten::gt.Tensor(Tensor self, Tensor other) -> Tensor", computeBcastShape},
    {"aten::ge.Tensor(Tensor self, Tensor other) -> Tensor", computeBcastShape},
    {"aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
     computeBcastShape},
    {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
     "values, Tensor indices)",
     computeReduceDimShape},
    {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
     computeSelectShape},
    {"immut::select(Tensor src, int dim, int index) -> Tensor",
     computeSelectShape},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     computeSliceShape},
    {"immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
     "end=None, SymInt step=1) -> Tensor",
     computeSliceShape},
    {"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
     computeUnsqueezeShape},
    {"immut::unsqueeze(Tensor self, int dim) -> Tensor", computeUnsqueezeShape},
    {"aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
     computeTransposeShape},
    {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
     computePermuteShape},
    {"aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
     computeReshapeShape},
    {"aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> "
     "Tensor(a)",
     computeExpandShape},
    {"aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)",
     computeAsShape},
    {"aten::repeat(Tensor self, SymInt[] repeats) -> Tensor",
     computeRepeatShape},
    {"aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
     computeIndexShape},
    {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", computeCatShape},
    {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", computeStackShape},
};

OperatorSet identicalShapeOps{
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=0) -> "
    "Tensor(a)",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    "aten::exp(Tensor self) -> Tensor",
    "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
    "aten::triu(Tensor self, int diagonal=0) -> Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
    "tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> Tensor",
    "immut::select_rev(Tensor self, Tensor src, int dim, int index) -> Tensor",
    "tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
    "SymInt? end=None, SymInt step=1) -> Tensor",
    "immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
    "SymInt? end=None, SymInt step=1) -> Tensor",
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
