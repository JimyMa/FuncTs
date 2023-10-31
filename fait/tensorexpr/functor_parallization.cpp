//
// Created by jimyma on 1/31/23.
//

#include "tensorexpr/functor_parallization.h"

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_cloner.h>

using namespace torch::jit::tensorexpr;

ExprPtr FunctorParallizationShapeMutator::mutate(VarPtr v) {
  if (dims_map_.count(v)) {
    return dims_map_[v][idx_].node();
  }
  return v;
}

ExprPtr FunctorParallizationShapeMutator::mutate(BufPtr v) {
  auto buf_op = v;
  std::vector<ExprPtr> new_dims;
  for (auto dim : buf_op->dims()) {
    new_dims.emplace_back(dim->accept_mutator(this));
  }
  buf_op->set_dims(new_dims);

  std::vector<ExprPtr> new_strides;
  for (auto &stride : buf_op->strides())
    new_strides.push_back(stride->accept_mutator(this));
  buf_op->set_strides(new_strides);

  return buf_op;
}

ExprPtr FunctorParallizationMutator::mutate(VarPtr v) {
  if (var_args_map_.count(v)) {
    return var_args_map_[v][idx_].node();
  }
  return v;
}

// ExprPtr
ExprPtr FunctorParallizationMutator::mutate(BufPtr v) {
  if (buf_ret_map_.count(v)) {
    IRCloner cloner;
    auto parallel_buf = buf_ret_map_[v];
    auto buf_op = to<Buf>(parallel_buf[idx_].node()->accept_mutator(&cloner));
    std::vector<ExprPtr> new_dims;
    for (auto dim : buf_op->dims()) {
      new_dims.emplace_back(dim->accept_mutator(this));
    }
    buf_op->set_dims(new_dims);

    std::vector<ExprPtr> new_strides;
    for (auto &stride : buf_op->strides())
      new_strides.push_back(stride->accept_mutator(this));
    buf_op->set_strides(new_strides);

    return buf_op;
  }

  if (buf_args_map_.count(v)) {
    IRCloner cloner;
    auto parallel_buf = buf_args_map_[v];
    auto buf_op = to<Buf>(parallel_buf[idx_].node()->accept_mutator(&cloner));
    std::vector<ExprPtr> new_dims;
    for (auto dim : buf_op->dims()) {
      new_dims.emplace_back(dim->accept_mutator(this));
    }
    buf_op->set_dims(new_dims);

    std::vector<ExprPtr> new_strides;
    for (auto &stride : buf_op->strides())
      new_strides.push_back(stride->accept_mutator(this));
    buf_op->set_strides(new_strides);

    return buf_op;
  }

  return v;
}

StmtPtr FunctorParallization::Parallel_functor(
    StmtPtr s, int64_t degree, VarPtr iter_var,
    std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
    std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map,
    std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map,
    std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map) {
  IRCloner clone;
  auto top_for = to<For>(s);
  auto functor_stmt = top_for->body();
  std::vector<StmtPtr> parallel_stmt;
  for (int i = 0; i < degree; i++) {
    IRCloner clone;
    FunctorParallizationMutator functor_parallel_mutator(
        degree, i, buf_args_map, var_args_map, buf_ret_map);
    FunctorParallizationShapeMutator functor_parallel_shape_mutator(degree, i,
                                                                    dims_map);
    parallel_stmt.emplace_back(
        functor_stmt->accept_mutator(&clone)
            ->accept_mutator(&functor_parallel_shape_mutator)
            ->accept_mutator(&functor_parallel_mutator));
  }
  auto if_then_else = parallel_stmt[0];
  for (int i = 1; i < degree; i++) {
    if_then_else = Cond::make(ExprHandle(iter_var) == LongImm::make(i),
                              parallel_stmt[i], if_then_else);
  }
  top_for->set_body(if_then_else);
  return top_for;
}