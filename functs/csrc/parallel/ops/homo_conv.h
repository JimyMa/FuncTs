#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace c10 {
namespace functs_parallel {

static const Symbol functs_parallel_ns =
    Symbol::fromQualString("namespaces::functs_parallel");
static auto HomoConv = Symbol::fromQualString("functs_parallel::homo_conv");

} // namespace functs_parallel

namespace attr {} // namespace attr

} // namespace c10
