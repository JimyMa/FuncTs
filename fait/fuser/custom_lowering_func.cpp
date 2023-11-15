//
// Created by jimyma on 2/10/23.
//

#include <utility>

#include "functs/csrc/jit/ir/symbol_ext.h"
#include "fuser/nnc_func.h"
#include "lowering_utils.h"
#include "tssa_set_ops.h"
#include "util/logging.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static Tensor computeTensor(CUSTOM_LOWERING_PARAMS) {
  return Compute("tensor", outShape, [&](ParameterList &axes) {
    return GET_ANY_SCALAR_EXPR_AT(0);
  });
}

template <class T>
static CustomLoweringFunction getComputeFillConst(T val,
                                                  const std::string &name) {
  return [=](CUSTOM_LOWERING_PARAMS) {
    return Compute("fill_" + name, outShape, [&](ParameterList &axes) {
      return ExprHandle(getImmediateByType(Dtype(outDtype), val));
    });
  };
}

static Tensor computeArange(CUSTOM_LOWERING_PARAMS) {
  return Compute("arange", outShape, [&](const VarHandle &i) {
    auto start = GET_INT_EXPR_AT(0);
    return Cast::make(Dtype(outDtype), start + i);
  });
}

static Tensor computeTriu(CUSTOM_LOWERING_PARAMS) {
  return Compute("triu", outShape, [&](ParameterList &axes) {
    auto self = GET_BUF_AT(0);
    auto diagonal = GET_INT_EXPR_AT(1);
    auto row = *(axes.end() - 2), col = axes.back();
    return IfThenElse::make((col - row) >= diagonal, self.load(axes),
                            ExprHandle(immLike(self, 0)));
  });
}

static ExprHandle loadBcast(const BufHandle &srcBuf, const ShapeVec &dstShape,
                            ParameterList &axes) {
  auto srcRank = srcBuf.dims().size(), dstRank = dstShape.size();
  std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
  loadAxes.erase(loadAxes.begin(), loadAxes.begin() + dstRank - srcRank);
  for (auto i : c10::irange(srcRank)) {
    loadAxes[i] =
        IfThenElse::make(srcBuf.dim(i) == int64_t(1), int64_t(0), loadAxes[i]);
  }
  return srcBuf.load(loadAxes);
}

template <class BinOp>
static CustomLoweringFunction getComputeBinaryBcast(BinOp &&op,
                                                    const std::string &name) {
  return [=](CUSTOM_LOWERING_PARAMS) {
    return Compute(name + "_bcast", outShape, [&](ParameterList &axes) {
      auto lhs = loadBcast(GET_BUF_AT(0), outShape, axes);
      auto rhs = loadBcast(GET_BUF_AT(1), outShape, axes);
      return op(lhs, rhs);
    });
  };
}

static ExprHandle max(const ExprHandle &lhs, const ExprHandle &rhs) {
  return Max::make(lhs, rhs, true);
}

static ExprHandle min(const ExprHandle &lhs, const ExprHandle &rhs) {
  return Min::make(lhs, rhs, true);
}

static ExprHandle lowestVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name)                                           \
  case ScalarType::Name:                                                       \
    return ExprHandle(std::numeric_limits<Type>::lowest());
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
  default:
    throw unsupported_dtype();
  }
}

static std::vector<VarHandle>
moveReduceAxisToDim(const std::vector<VarHandle> &axes, int64_t dim,
                    bool keepdim) {
  auto result = axes;
  auto reduceAxis = axes.back();
  result.pop_back();
  if (keepdim)
    result.at(dim) = reduceAxis;
  else
    result.insert(result.begin() + dim, reduceAxis);
  return result;
}

static Tensor computeMaxDim(CUSTOM_LOWERING_PARAMS) {
  // Process arguments
  auto self = GET_BUF_AT(0);
  auto selfShape = self.dims();
  auto rank = selfShape.size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += rank;
  auto keepdim = GET_BOOL_CONST_AT(2);
  auto reduceDim = selfShape.at(dim);

  // Compute reduction
  return Reduce("max_dim", outShape, Maximum(lowestVal(outDtype)),
                [&](ParameterList &axes) {
                  return self.load(moveReduceAxisToDim(axes, dim, keepdim));
                },
                {reduceDim});
}

static Tensor computeSoftmax(CUSTOM_LOWERING_PARAMS) {
  // Process arguments
  auto self = GET_BUF_AT(0);
  auto selfShape = self.dims();
  auto rank = selfShape.size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0)
    dim += rank;

  // Compute reduced shape and axes
  auto reduceDim = selfShape.at(dim);
  auto reducedShape = selfShape;
  reducedShape.erase(reducedShape.begin() + dim);
  auto removeDimAxis = [&](ParameterList &axes) {
    std::vector<ExprHandle> result(axes.begin(), axes.end());
    result.erase(result.begin() + dim);
    return result;
  };

  // Define compute for softmax
  auto dimMax =
      Reduce("softmax_max", reducedShape, Maximum(lowestVal(outDtype)),
             [&](ParameterList &axes) {
               return self.load(moveReduceAxisToDim(axes, dim, false));
             },
             {reduceDim});
  auto expSubMax = Compute("softmax_exp", selfShape, [&](ParameterList &axes) {
    return exp(self.load(axes) - dimMax.load(removeDimAxis(axes)));
  });
  auto dimSum =
      Reduce("softmax_sum", reducedShape, Sum(),
             [&](ParameterList &axes) {
               return expSubMax.load(moveReduceAxisToDim(axes, dim, false));
             },
             {reduceDim});
  auto result = Compute("softmax", selfShape, [&](ParameterList &axes) {
    return expSubMax.load(axes) / dimSum.load(removeDimAxis(axes));
  });

  std::vector<StmtPtr> stmts{dimMax.stmt(), expSubMax.stmt(), dimSum.stmt(),
                             result.stmt()};
  return Tensor(result.buf(), Block::make(std::move(stmts)));
}

static Tensor computeSelect(CUSTOM_LOWERING_PARAMS) {
  return Compute("select", outShape, [&](ParameterList &axes) {
    auto src = GET_BUF_AT(0);
    auto rank = src.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0)
      dim += rank;
    auto dimSize = src.dims().at(dim);
    auto idx = GET_INT_EXPR_AT(2);
    idx = IfThenElse::make(idx >= int64_t(0), idx, idx + dimSize);

    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx.insert(output_idx.begin() + dim, idx);

    return src.load(output_idx);
  });
}

static Tensor computeSlice(CUSTOM_LOWERING_PARAMS) {
  return Compute("slice", outShape, [&](ParameterList &axes) {
    // Source tensor
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
      start = int64_t(0);
    else
      start = getScalarExpr<int64_t>(startVal, valueToExpr);
    start = IfThenElse::make(start >= int64_t(0),
                             Min::make(start, dimSize, true), start + dimSize);

    // Step
    int64_t step = GET_INT_CONST_AT(4);
    // Source indices
    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx[dim] = start + LongImm::make(step) * output_idx[dim];

    return src.load(output_idx);
  });
}

static Tensor computeUnsqueeze(CUSTOM_LOWERING_PARAMS) {
  return Compute("unsqueeze", outShape, [&](ParameterList &axes) {
    auto self = GET_BUF_AT(0);
    auto rank = self.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0)
      dim += rank + 1;
    auto loadAxes = axes;
    loadAxes.erase(loadAxes.begin() + dim);
    return self.load(loadAxes);
  });
}

static Tensor computeAssign(CUSTOM_LOWERING_PARAMS) {
  return Compute("assign", outShape, [&](ParameterList &axes) {
    // Remove front axes
    auto self = GET_BUF_AT(1);
    return self.load(axes);
  });
}

static Tensor computeClone(CUSTOM_LOWERING_PARAMS) {
  return Compute("clone", outShape, [&](ParameterList &axes) {
    // Remove front axes
    auto self = GET_BUF_AT(0);
    return self.load(axes);
  });
}

static Tensor computeRepeat(CUSTOM_LOWERING_PARAMS) {
  return Compute("repeat", outShape, [&](ParameterList &axes) {
    // Remove front axes
    auto self = GET_BUF_AT(0);
    auto inShape = self.dims();
    auto inRank = inShape.size(), outRank = outShape.size();
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    loadAxes.erase(loadAxes.begin(), loadAxes.begin() + outRank - inRank);

    // Update load axes
    for (auto i : c10::irange(inRank)) {
      const auto &axis = loadAxes[i];
      loadAxes[i] =
          IfThenElse::make(inShape[i] == outShape[i], axis, axis % inShape[i]);
    }

    return self.load(loadAxes);
  });
}

static Tensor computeIndex(CUSTOM_LOWERING_PARAMS) {
  return Compute("index", outShape, [&](ParameterList &axes) {
    // Process inputs
    auto self = GET_BUF_AT(0);
    auto indexBuf = GET_BUF_LIST_AT(1).front();
    auto indexRank = indexBuf.dims().size();

    // Load index
    std::vector<VarHandle> indexAxes(axes.begin(), axes.begin() + indexRank);
    auto index = indexBuf.load(indexAxes);

    // Select `self` at dim 0 with loaded index
    std::vector<ExprHandle> selfAxes(axes.begin() + indexRank, axes.end());
    selfAxes.insert(selfAxes.begin(), index);

    return self.load(selfAxes);
  });
}

static Tensor computeCat(CUSTOM_LOWERING_PARAMS) {
  return Compute("cat", outShape, [&](ParameterList &axes) {
    // Process dimension
    auto bufs = GET_BUF_LIST_AT(0);
    auto inRank = bufs.front().dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0)
      dim += inRank;

    // Compute section index range
    std::vector<ExprHandle> indices(bufs.size() + 1);
    indices[0] = int64_t(0);
    for (auto i : c10::irange(bufs.size()))
      indices[i + 1] = indices[i] + bufs[i].dim(dim);

    // Switch buffers according to index range at concatenation axis
    auto dimAxis = axes[dim];
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    ExprHandle result(getImmediateByType(bufs.front().dtype(), 0));
    for (int64_t i = bufs.size() - 1; i >= 0; i--) {
      auto bufLoadAxes = loadAxes;
      bufLoadAxes[dim] = dimAxis - indices[i];
      result = IfThenElse::make(ExprHandle(dimAxis) < indices[i + 1],
                                bufs[i].load(bufLoadAxes), result);
    }

    return result;
  });
}

static Tensor computeStack(CUSTOM_LOWERING_PARAMS) {
  return Compute("stack", outShape, [&](ParameterList &axes) {
    // Process dimension
    auto bufs = GET_BUF_LIST_AT(0);
    auto inRank = bufs.front().dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0)
      dim += inRank + 1;

    // Switch buffers according to dim axis
    auto dimAxis = axes[dim];
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    loadAxes.erase(loadAxes.begin() + dim);
    ExprHandle result(getImmediateByType(bufs.front().dtype(), 0));
    for (int64_t i = bufs.size() - 1; i >= 0; i--) {
      result = IfThenElse::make(ExprHandle(dimAxis) == i,
                                bufs[i].load(loadAxes), result);
    }

    return result;
  });
}

using EwiseFunc = std::function<ExprHandle(const ExprHandle &)>;
#define EWISE_FUNC_CREATOR_PARAMS Node *node, const ValueExprMap &valueToExpr
using EwiseFuncCreator = std::function<EwiseFunc(EWISE_FUNC_CREATOR_PARAMS)>;

static EwiseFunc getClamp(EWISE_FUNC_CREATOR_PARAMS) {
  return [node, &valueToExpr](const ExprHandle &src) {
    auto result = src;
    if (node->input(1)->type()->kind() != TypeKind::NoneType)
      result = Max::make(
          result, Cast::make(src.dtype(), GET_ANY_SCALAR_EXPR_AT(1)), true);
    if (node->input(2)->type()->kind() != TypeKind::NoneType)
      result = Min::make(
          result, Cast::make(src.dtype(), GET_ANY_SCALAR_EXPR_AT(2)), true);
    return result;
  };
};

static OperatorMap<EwiseFuncCreator> ewiseExprCreators{

    {"aten::sigmoid(Tensor self) -> Tensor",
     [](EWISE_FUNC_CREATOR_PARAMS) { return sigmoid; }},
    {"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
     getClamp},
};

static std::vector<EwiseFunc>
findEwiseFuncsForSetSrc(Value *self, Value *src, Symbol viewSym,
                        const ValueExprMap &valueToExpr) {
  // Trace src to self
  Value *curVal = src;
  std::list<EwiseFunc> funcList;

  while (true) {
    // Check definition of value
    auto node = curVal->node();

    // If view symbol is encountered, check if it operates on self
    if (node->kind() == viewSym) {
      if (node->input(0) == self)
        break;
      else
        return {};
    }

    // Not the target view symbol, check if we can create an element-wise
    // expression
    auto op = node->maybeOperator();
    if (!op)
      return {};
    auto exprCreator = ewiseExprCreators.find(*op);
    if (!exprCreator)
      return {};
    funcList.push_front((*exprCreator)(node, valueToExpr));

    // Move to its first input
    curVal = node->input(0);
  }

  return {funcList.begin(), funcList.end()};
}

static Tensor computeSelectSet(CUSTOM_LOWERING_PARAMS) {
  return Compute("select_set", outShape, [&](ParameterList &axes) {
    auto self = GET_BUF_AT(0);
    auto rank = self.dims().size();
    auto src = GET_BUF_AT(1);
    auto dim = GET_INT_CONST_AT(2);
    if (dim < 0)
      dim += rank;
    auto dimSize = self.dims().at(dim);
    auto idx = GET_INT_EXPR_AT(3);
    idx = IfThenElse::make(idx >= int64_t(0), idx, idx + dimSize);

    std::vector<ExprHandle> srcAxes(axes.begin(), axes.end());
    srcAxes.erase(srcAxes.begin() + dim);
    auto cond =
        CompareSelect::make(axes[dim], idx, src.load(srcAxes), self.load(axes),
                            CompareSelectOperation::kEQ);

    return cond;
  });
}

static Tensor computeSliceSet(CUSTOM_LOWERING_PARAMS) {
  return Compute("slice_set", outShape, [&](ParameterList &axes) {
    // Tensor
    auto self = GET_BUF_AT(0);
    auto rank = self.dims().size();
    auto src = GET_BUF_AT(1);
    auto dim = GET_INT_CONST_AT(2);
    if (dim < 0)
      dim += rank;
    auto dimSize = self.dims().at(dim);

    // Start
    auto startVal = node->input(3);
    ExprHandle start;
    if (startVal->type()->kind() == TypeKind::NoneType)
      start = int64_t(0);
    else
      start = getScalarExpr<int64_t>(startVal, valueToExpr);
    // start = IfThenElse::make(start >= int64_t(0),
    //                          Min::make(start, dimSize, true), start +
    //                          dimSize);

    // End
    auto endVal = node->input(4);
    ExprHandle end;
    if (endVal->type()->kind() == TypeKind::NoneType)
      end = dimSize;
    else
      end = getScalarExpr<int64_t>(endVal, valueToExpr);
    // end = IfThenElse::make(end >= int64_t(0), Min::make(end, dimSize, true),
    //                        end + dimSize);

    // Step
    int64_t step = GET_INT_CONST_AT(5);

    // Setter axes
    std::vector<ExprHandle> srcAxes(axes.begin(), axes.end());
    auto dimAxis = axes[dim];
    srcAxes[dim] = (axes[dim] - start) / step;

    // See if we can create an elementwise pipeline for source values
    auto srcElem = src.load(srcAxes);
    auto ewiseFuncs = findEwiseFuncsForSetSrc(
        node->input(0), node->input(1), c10::immutable::Slice, valueToExpr);

    if (!ewiseFuncs.empty()) {
      srcElem = self.load(axes);
      for (auto &func : ewiseFuncs)
        srcElem = func(srcElem);
    }

    // Select elements
    auto notSet = (dimAxis < start) || (dimAxis >= end) ||
                  ((dimAxis - start) % step != int64_t(0));
    auto result = IfThenElse::make(notSet, self.load(axes), srcElem);

    return result;
  });
}

static auto _tssaSetOps = registerTssaSetOps();

OperatorMap<CustomLoweringFunction> customLoweringFuncs{
    {"aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
     computeClone},
    {"immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
     computeAssign},
    {"aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
     "bool requires_grad=False) -> Tensor",
     computeTensor},
    {"aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
     "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
     getComputeFillConst(0, "zeros")},
    {"aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
     "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
     "Tensor",
     computeArange},
    {"aten::triu(Tensor self, int diagonal=0) -> Tensor", computeTriu},
    {"aten::maximum(Tensor self, Tensor other) -> Tensor",
     getComputeBinaryBcast(max, "max")},
    {"aten::minimum(Tensor self, Tensor other) -> Tensor",
     getComputeBinaryBcast(min, "min")},
    {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
     "values, Tensor indices)",
     computeMaxDim},
    {"aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> "
     "Tensor",
     computeSoftmax},
    {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
     computeSelect},
    {"immut::select(Tensor self, int dim, int index) -> Tensor self",
     computeSelect},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     computeSlice},
    {"immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
     "end=None, SymInt step=1) -> Tensor",
     computeSlice},
    {"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", computeUnsqueeze},
    {"immut::unsqueeze(Tensor self, int dim) -> Tensor", computeUnsqueeze},
    {"aten::repeat(Tensor self, SymInt[] repeats) -> Tensor", computeRepeat},
    {"aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
     computeIndex},
    {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", computeCat},
    {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", computeStack},
    {"tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> Tensor",
     computeSelectSet},
    {"immut::select_rev(Tensor self, Tensor src, int dim, int index) -> Tensor",
     computeSelectSet},
    {"tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor",
     computeSliceSet},
    {"immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor",
     computeSliceSet},

};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
