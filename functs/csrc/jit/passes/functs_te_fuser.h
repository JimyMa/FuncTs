#pragma once
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
void FuncTsFuseTensorExprs(std::shared_ptr<Graph>& graph, size_t min_group_size,
                           bool add_composed_op, bool fuse_to_dynamic_shapes);
}
}  // namespace torch
