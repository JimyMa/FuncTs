#include <cstdint>
#include <functs/csrc/jit/tensorexpr/nnc_ext.h>

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

Tensor
computeImmutAssign(const std::vector<ArgValue> &inputValues,
                   const std::vector<ExprHandle> &outputShape,
                   c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("assign", outputShape, [&](ParameterList &axes) {
    auto src = c10::get<torch::jit::tensorexpr::BufHandle>(inputValues[0]);
    return src.load(axes);
  });
}

Tensor computeImmutSlice(const std::vector<ArgValue> &inputValues,
                         const std::vector<ExprHandle> &outputShape,
                         c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute(
      "immut_slice", outputShape, outputStrides, [&](ParameterList &axes) {
        // Source tensor
        auto src = c10::get<torch::jit::tensorexpr::BufHandle>(inputValues[0]);
        auto rank = src.dims().size();

        auto dim = c10::get<int64_t>(inputValues[1]);
        if (dim < 0)
          dim += rank;
        auto dimSize = src.dims().at(dim);
        // Start
        ExprHandle start = constant(inputValues[2]);
        // IfThenElse::make(start >= int64_t(0), Min::make(start, dimSize,
        // true),
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
computeImmutSelect(const std::vector<ArgValue> &inputValues,
                   const std::vector<ExprHandle> &outputShape,
                   c10::optional<std::vector<ExprHandle>> outputStrides) {
  return Compute("select", outputShape, [&](ParameterList &axes) {
    auto src = c10::get<torch::jit::tensorexpr::BufHandle>(inputValues[0]);
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

} // namespace tensorexpr
} // namespace jit
} // namespace torch
