#include <cstdint>
#include <functs/csrc/jit/tensorexpr/nnc_ext.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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
        auto startVal = c10::get<int64_t>(inputValues[2]);
        ExprHandle start;
        IfThenElse::make(start >= int64_t(0), Min::make(start, dimSize, true),
                         start + dimSize);

        // Step
        int64_t step = c10::get<int64_t>(inputValues[2]);
        // Source indices
        std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
        output_idx[dim] = start + LongImm::make(step) * output_idx[dim];

        return src.load(output_idx);
      });
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
