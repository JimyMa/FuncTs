#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include "refine_types.h"

namespace torch {
namespace jit {

void FuseOps(const std::shared_ptr<Graph> &graph, ValueTypeMap &refinedTypes);

Node *commitFusion(Node *head, Node *tail, Graph *graph,
                   ValueTypeMap &refinedTypes);

}  // namespace jit
}  // namespace torch
