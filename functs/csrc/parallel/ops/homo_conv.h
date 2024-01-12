#pragma once

#include <torch/extension.h>
#include "torch/csrc/jit/ir/ir.h"

namespace c10 {
namespace functs_parallel {

static const Symbol functs_parallel_ns =
    Symbol::fromQualString("namespaces::functs_parallel");
static auto HomoConv = Symbol::fromQualString("functs_parallel::homo_conv");

} // namespace functs_parallel

namespace attr {} // namespace attr

} // namespace c10
namespace torch {
namespace jit {
void homo_invoke(
    const std::vector<at::Tensor>& feats,
    const std::vector<at::Tensor>& outs,
    const std::vector<at::Tensor>& weight,
    const std::vector<at::Tensor>& bias);
} // namespace jit
} // namespace torch
