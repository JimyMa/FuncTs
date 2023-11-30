#include <functs/csrc/jit/tensorexpr/nnc_ext.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <cstdint>

namespace torch {
namespace jit {
namespace tensorexpr {

ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes) {
  return b.load(computeIndicesToBroadcast(axes, b.dims()));
}

ExprHandle constant(const ArgValue& v) {
  if (auto s = c10::get_if<tensorexpr::VarHandle>(&v)) {
    return *s;
  } else if (auto d = c10::get_if<double>(&v)) {
    return DoubleImm::make(*d);
  } else if (auto i = c10::get_if<int64_t>(&v)) {
    return LongImm::make(*i);
  } else if (auto b = c10::get_if<bool>(&v)) {
    return BoolImm::make(*b);
  } else if (c10::get_if<ArgNone>(&v)) {
    // This is just a placeholder so we don't throw.  None-handling
    // is operator-specific and should be handled properly in
    // the operator-specific lowering code.
    return IntImm::make(0);
  } else {
    throw unsupported_dtype("Trying to convert unsupported dtype to constant");
  }
}

std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<ExprHandle>& outputAxes,
    const std::vector<ExprHandle>& inputSizes) {
  if (outputAxes.size() < inputSizes.size()) {
    throw malformed_input("Cannot broadcast to a lower rank tensor");
  }
  std::vector<ExprHandle> bcast;
  auto axisIt = outputAxes.rbegin();
  auto sizeIt = inputSizes.rbegin();
  while (sizeIt != inputSizes.rend()) {
    auto const& size = intValue(*sizeIt);
    if (size && *size == 1) {
      bcast.emplace_back(LongImm::make(0));
    } else {
      bcast.emplace_back(*axisIt);
    }
    ++axisIt;
    ++sizeIt;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

ExprHandle scalarOrConstant(const ArgValue& v) {
  if (auto vh = c10::get_if<VarHandle>(&v)) {
    return *vh;
  }
  return constant(v);
}

Tensor computeClone(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("clone", outputShape, [&](ParameterList& axes) {
    auto src = c10::get<BufHandle>(inputValues[0]);
    return src.load(axes);
  });
}

Tensor computeImmutAssign(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("assign", outputShape, [&](ParameterList& axes) {
    auto src = c10::get<BufHandle>(inputValues[1]);
    return src.load(axes);
  });
}

Tensor computeImmutSlice(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute(
      "immut_slice", outputShape, outputStrides, [&](ParameterList& axes) {
        // Source tensor
        auto src = c10::get<BufHandle>(inputValues[0]);
        auto rank = src.dims().size();

        auto dim = c10::get<int64_t>(inputValues[1]);
        if (dim < 0)
          dim += rank;
        auto dimSize = src.dims().at(dim);
        // Start
        ExprHandle start = scalarOrConstant(inputValues[2]);
        // IfThenElse::make(start >= int64_t(0), Min::make(start,
        // dimSize, true),
        //                  start + dimSize);

        // Step
        auto step = scalarOrConstant(inputValues[4]);
        // Source indices
        std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
        output_idx[dim] = start + step * output_idx[dim];

        return src.load(output_idx);
      });
}

Tensor computeImmutSliceRev(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute(
      "slice_rev", outputShape, outputStrides, [&](ParameterList& axes) {
        // Tensor
        auto self = c10::get<BufHandle>(inputValues[0]);
        auto rank = self.dims().size();
        auto src = c10::get<BufHandle>(inputValues[1]);
        auto dim = c10::get<int64_t>(inputValues[2]);
        if (dim < 0)
          dim += rank;
        auto dimSize = self.dims().at(dim);

        // Start
        ExprHandle start;
        if (c10::get_if<ArgNone>(&inputValues[3]))
          start = int64_t(0);
        else
          start = scalarOrConstant(inputValues[3]);
        start = IfThenElse::make(
            start >= int64_t(0),
            Min::make(start, dimSize, true),
            start + dimSize);

        // End
        ExprHandle end;
        if (c10::get_if<ArgNone>(&inputValues[4]))
          end = dimSize;
        else
          end = scalarOrConstant(inputValues[4]);
        end = IfThenElse::make(
            end >= int64_t(0), Min::make(end, dimSize, true), end + dimSize);

        // Step
        auto step = constant(inputValues[5]);

        // Setter axes
        std::vector<ExprHandle> srcAxes(axes.begin(), axes.end());
        auto dimAxis = axes[dim];
        srcAxes[dim] = (axes[dim] - start) / step;

        // See if we can create an elementwise pipeline for source values
        auto srcElem = src.load(srcAxes);

        // Select elements
        auto notSet = (dimAxis < start) || (dimAxis >= end) ||
            ((dimAxis - start) % step != int64_t(0));
        auto result = IfThenElse::make(notSet, self.load(axes), srcElem);
        return result;
      });
}

Tensor computeImmutSelect(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("select", outputShape, [&](ParameterList& axes) {
    auto src = c10::get<BufHandle>(inputValues[0]);
    auto rank = src.dims().size();
    int64_t dim = c10::get<int64_t>(inputValues[1]);

    if (dim < 0)
      dim += rank;

    auto idx = scalarOrConstant(inputValues[2]);

    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx.insert(output_idx.begin() + dim, idx);

    return src.load(output_idx);
  });
}

Tensor computeImmutSelectRev(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("select_rev", outputShape, [&](ParameterList& axes) {
    auto self = c10::get<BufHandle>(inputValues[0]);
    auto rank = self.dims().size();
    auto src = c10::get<BufHandle>(inputValues[1]);
    auto dim = c10::get<int64_t>(inputValues[2]);
    if (dim < 0)
      dim += rank;
    auto dimSize = self.dims().at(dim);
    auto idx = constant(inputValues[3]);
    idx = IfThenElse::make(idx >= int64_t(0), idx, idx + dimSize);

    std::vector<ExprHandle> srcAxes(axes.begin(), axes.end());
    srcAxes.erase(srcAxes.begin() + dim);
    auto cond = CompareSelect::make(
        axes[dim],
        idx,
        src.load(srcAxes),
        self.load(axes),
        CompareSelectOperation::kEQ);
    return cond;
  });
}

Tensor computeImmutUnsqueeze(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute(
      "unsqueeze", outputShape, outputStrides, [&](ParameterList& axes) {
        auto self = c10::get<BufHandle>(inputValues[0]);
        auto rank = self.dims().size();
        auto dim = c10::get<int64_t>(inputValues[1]);
        if (dim < 0)
          dim += rank + 1;
        auto loadAxes = axes;
        loadAxes.erase(loadAxes.begin() + dim);
        return self.load(loadAxes);
      });
}

Tensor computeImmutView(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  auto A = c10::get<BufHandle>(inputValues[0]);
  if (A.ndim() == 0) {
    return Compute(
        "aten_view", outputShape, [&](const std::vector<VarHandle>& axes) {
          std::vector<ExprHandle> empty_indices;
          return A.load(empty_indices);
        });
  }
  return Compute(
      "aten_reshape", outputShape, [&](const std::vector<VarHandle>& axes) {
        std::vector<VarHandle> new_axes;
        assert(outputShape.size() == axes.size());
        /*
        Example for the index transformation. Assume we have a tensor A and
        its view B:
          A.size() = [6,2,3]
          B = A.view(2,1,9,1,2)

        In TE IR we would want to represent B as the following loopnest:
          for (i1 in 0..2)
            for (i2 in 0..1)
              for (i3 in 0..9)
                for (i4 in 0..1)
                  for (i5 in 0..2)
                    idx = i5 + i4*2 + i3*2 + i2*18 + i1*18
                    B[i1,i2,i3,i4,i5] = A[idx/(3*2), (idx/3)%2, idx%3]
        */
        std::vector<ExprPtr> dims, indices;
        for (size_t idx = 0; idx < outputShape.size(); idx++) {
          dims.push_back(outputShape[idx].node());
          indices.push_back(axes[idx].node());
        }

        auto ndim = dims.size();
        std::vector<ExprPtr> strides(ndim);
        strides[ndim - 1] = immLike(dims[ndim - 1], 1);
        for (size_t i = 1; i < ndim; i++) {
          strides[ndim - 1 - i] = alloc<Mul>(strides[ndim - i], dims[ndim - i]);
        }

        ExprHandle flat_idx = ExprHandle(flatten_index(dims, indices, strides));
        std::vector<ExprHandle> orig_buf_indexes(A.ndim(), ExprHandle(0));
        ExprHandle stride = ExprHandle(immLike(flat_idx, 1));
        for (size_t idx = 0; idx < A.ndim(); idx++) {
          size_t dim_idx = A.ndim() - idx - 1;
          // We don't need to generate mod-div for the first dimension -
          // ideally IRSimplifier would get rid of that for us, but for now
          // let's just avoid generating it in the first place.
          if (dim_idx > 0) {
            orig_buf_indexes[dim_idx] = flat_idx / stride % A.dim(dim_idx);
          } else {
            orig_buf_indexes[dim_idx] = flat_idx / stride;
          }
          // In the example above the stride is initially 1 for dim_idx = 2,
          // then it's 3 for dim_idx = 1, and then it's 3*2 for dim_idx = 0.
          stride = stride * A.dim(dim_idx);
        }
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        return A.load(orig_buf_indexes);
      });
}

Tensor computeImmutRepeat(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("repeat", outputShape, [&](ParameterList& axes) {
    // Remove front axes
    auto self = c10::get<BufHandle>(inputValues[0]);
    auto inShape = self.dims();
    auto inRank = inShape.size(), outRank = outputShape.size();
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    loadAxes.erase(loadAxes.begin(), loadAxes.begin() + outRank - inRank);

    // Update load axes
    for (auto i : c10::irange(inRank)) {
      const auto& axis = loadAxes[i];
      loadAxes[i] = IfThenElse::make(
          inShape[i] == outputShape[i], axis, axis % inShape[i]);
    }

    return self.load(loadAxes);
  });
}

TORCH_API Tensor computeImmutPermute(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  auto A = c10::get<BufHandle>(inputValues[0]);
  // Trivial case of 0-dim tensors: just a copy of the input
  if (A.ndim() == 0) {
    auto tensor = Compute(
        "permute",
        outputShape,
        outputStrides,
        [&](const std::vector<VarHandle>& axes) {
          std::vector<ExprHandle> empty_indices;
          return A.load(empty_indices);
        });
    if (A.node()->qscale()) {
      tensor.buf()->set_qscale(A.node()->qscale());
      tensor.buf()->set_qzero(A.node()->qzero());
    }
    return tensor;
  }
  auto permute_dims = c10::get<IntList>(inputValues[1]);
  auto tensor = Compute(
      "aten_permute", outputShape, [&](const std::vector<VarHandle>& axes) {
        std::vector<VarHandle> new_axes;
        new_axes.resize(axes.size());
        assert(permute_dims.size() == axes.size());
        for (unsigned i = 0; i < axes.size(); i++) {
          auto new_dim = at::maybe_wrap_dim(permute_dims[i], A.ndim());
          new_axes[new_dim] = axes[i];
        }
        return A.load(new_axes);
      });
  if (A.node()->qscale()) {
    tensor.buf()->set_qscale(A.node()->qscale());
    tensor.buf()->set_qzero(A.node()->qzero());
  }
  return tensor;
}

TORCH_API Tensor computeImmutExpand(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  auto A = c10::get<BufHandle>(inputValues[0]);
  return Compute(
      "expand", outputShape, [&](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        return broadcast(A, indices);
      });
}

TORCH_API Tensor computeImmutIndex(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("index", outputShape, [&](const std::vector<VarHandle>& axes) {
    // Process inputs
    auto self = c10::get<BufHandle>(inputValues[0]);
    auto indexBuf = c10::get<BufList>(inputValues[1]).front();
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

TORCH_API Tensor computeTensor(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("tensor", outputShape, [&](ParameterList& axes) {
    return scalarOrConstant(inputValues[0]);
  });
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
