#pragma once

#include <ATen/Context.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/jit_type_base.h>
#include <ATen/ops/allclose.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
#include <memory>
#include <unordered_map>

#include "passes/canonicalize.h"
#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/op_stat.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/te_op.h"
#include "passes/tensor_ssa.h"
#include "passes/type_utils.h"
#include "passes/unroll_loops.h"
#include "passes/validate_graph.h"
#include "run_utils.h"
#include "util/logging.h"
#include "util/rand.h"

namespace torch {
namespace jit {

void FaitPipeline(std::shared_ptr<Graph>, std::vector<c10::TypePtr>);
void FaitGetRefineType(
    std::shared_ptr<Graph>,
    std::vector<c10::TypePtr>,
    std::unordered_map<Value*, TypePtr>& refinedTypes);

} // namespace jit
} // namespace torch
