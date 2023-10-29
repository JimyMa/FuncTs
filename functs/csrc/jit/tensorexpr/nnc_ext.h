#pragma once
#include <cstdint>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor computeImmutAssign(
    const std::vector<ArgValue> &inputValues,
    const std::vector<ExprHandle> &outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutSelect(
    const std::vector<ArgValue> &inputValues,
    const std::vector<ExprHandle> &outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutSlice(
    const std::vector<ArgValue> &inputs,
    const std::vector<ExprHandle> &outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
