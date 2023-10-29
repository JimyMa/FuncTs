#include <cstdint>
#include <functs/csrc/jit/tensorexpr/nnc_ext.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch {
namespace jit {
namespace tensorexpr {

ExprHandle constant(const ArgValue &v) {
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

Tensor computeClone(const std::vector<ArgValue> &inputValues,
                    const std::vector<ExprHandle> &outputShape,
                    c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("clone", outputShape, [&](ParameterList &axes) {
    auto src = c10::get<BufHandle>(inputValues[0]);
    return src.load(axes);
  });
}

Tensor
computeImmutAssign(const std::vector<ArgValue> &inputValues,
                   const std::vector<ExprHandle> &outputShape,
                   c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("assign", outputShape, [&](ParameterList &axes) {
    auto src = c10::get<BufHandle>(inputValues[1]);
    return src.load(axes);
  });
}

Tensor computeImmutSlice(const std::vector<ArgValue> &inputValues,
                         const std::vector<ExprHandle> &outputShape,
                         c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("immut_slice", outputShape, outputStrides,
                 [&](ParameterList &axes) {
                   // Source tensor
                   auto src = c10::get<BufHandle>(inputValues[0]);
                   auto rank = src.dims().size();

                   auto dim = c10::get<int64_t>(inputValues[1]);
                   if (dim < 0)
                     dim += rank;
                   auto dimSize = src.dims().at(dim);
                   // Start
                   ExprHandle start = constant(inputValues[2]);
                   // IfThenElse::make(start >= int64_t(0), Min::make(start,
                   // dimSize, true),
                   //                  start + dimSize);

                   // Step
                   auto step = constant(inputValues[4]);
                   // Source indices
                   std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
                   output_idx[dim] = start + step * output_idx[dim];

                   return src.load(output_idx);
                 });
}

Tensor
computeImmutSliceRev(const std::vector<ArgValue> &inputValues,
                     const std::vector<ExprHandle> &outputShape,
                     c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute(
      "slice_rev", outputShape, outputStrides, [&](ParameterList &axes) {
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
          start = constant(inputValues[3]);
        start =
            IfThenElse::make(start >= int64_t(0),
                             Min::make(start, dimSize, true), start + dimSize);

        // End
        ExprHandle end;
        if (c10::get_if<ArgNone>(&inputValues[4]))
          end = dimSize;
        else
          end = constant(inputValues[4]);
        end = IfThenElse::make(end >= int64_t(0), Min::make(end, dimSize, true),
                               end + dimSize);

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

Tensor
computeImmutSelect(const std::vector<ArgValue> &inputValues,
                   const std::vector<ExprHandle> &outputShape,
                   c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("select", outputShape, [&](ParameterList &axes) {
    auto src = c10::get<BufHandle>(inputValues[0]);
    auto rank = src.dims().size();
    int64_t dim = c10::get<int64_t>(inputValues[1]);

    if (dim < 0)
      dim += rank;

    auto idx = constant(inputValues[2]);

    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx.insert(output_idx.begin() + dim, idx);

    return src.load(output_idx);
  });
}

Tensor
computeImmutUnsqueeze(const std::vector<ArgValue> &inputValues,
                      const std::vector<ExprHandle> &outputShape,
                      c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("unsqueeze", outputShape, outputStrides,
                 [&](ParameterList &axes) {
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

} // namespace tensorexpr
} // namespace jit
} // namespace torch
