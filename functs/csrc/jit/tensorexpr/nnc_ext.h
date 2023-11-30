#pragma once
#include <c10/util/Optional.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>
#include <cstdint>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor computeImmutAssign(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeClone(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutSelect(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

Tensor computeImmutSelectRev(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutSlice(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutSliceRev(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutUnsqueeze(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutView(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutRepeat(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutPermute(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutExpand(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeImmutIndex(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

TORCH_API Tensor computeTensor(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    c10::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
