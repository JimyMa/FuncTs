//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_TE_OP_H
#define LONG_TAIL_TE_OP_H

#include "torch/csrc/jit/ir/ir.h"

namespace c10 {
namespace tssa {

static auto ParallelFunctor = Symbol::fromQualString("tssa::ParallelFunctor");

}  // namespace tssa

namespace attr {

static auto parallel_degree = Symbol::fromQualString("attr::parallel_degree");
static auto is_parallel_args = Symbol::fromQualString("attr::is_parallel_args");
static auto input_refine_types =
    Symbol::fromQualString("attr::input_refine_types");
static auto is_parallel_map = Symbol::fromQualString("attr::is_parallel_map");

}  // namespace attr

}  // namespace c10

namespace torch {
namespace jit {

void MapFunctorToParallelization(
    const std::shared_ptr<Graph> &graph,
    std::unordered_map<Value *, TypePtr> &refine_types);

void FusedOpToParallelization(
    const std::shared_ptr<Graph> &graph,
    std::unordered_map<Value *, TypePtr> &refine_types);

}  // namespace jit
}  // namespace torch

#endif  // LONG_TAIL_TE_OP_H
