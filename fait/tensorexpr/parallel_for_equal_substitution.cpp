#include "tensorexpr/parallel_for_equal_substitution.h"

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir.h>

#include "util/logging.h"

using namespace torch::jit::tensorexpr;

StmtPtr ParallelForEqualSubstitutionMutator::mutate(ForPtr v) {
  auto start_new = v->start()->accept_mutator(this);
  auto options = std::move(v->loop_options());
  auto body_new = v->body()->accept_mutator(this);
  if (options.is_gpu_block_index() && options.gpu_block_index() == 0)
    mutate_state_ = true;
  auto stop_new = v->stop()->accept_mutator(this);
  mutate_state_ = false;

  return alloc<For>(v->var(), start_new, stop_new, body_new, options);
}

StmtPtr ParallelForEqualSubstitutionMutator::mutate(BlockPtr v) {
  std::vector<StmtPtr> stmts_new;
  stmts_new.reserve(v->nstmts());
  for (StmtPtr stmt : *v) {
    stmts_new.push_back(stmt->accept_mutator(this));
  }
  return alloc<Block>(stmts_new);
}

ExprPtr ParallelForEqualSubstitutionMutator::mutate(CompareSelectPtr v) {
  auto ret_1 = v->ret_val1()->accept_mutator(this);
  auto ret_2 = v->ret_val2()->accept_mutator(this);
  if (mutate_state_) {
    return alloc<Max>(ret_1, ret_2, false);
  } else {
    return alloc<CompareSelect>(v->lhs(), v->rhs(), ret_1, ret_2,
                                v->compare_select_op());
  }
}

StmtPtr ParallelForEqualSubstitution::run(StmtPtr s) {
  ParallelForEqualSubstitutionMutator mutator;
  s->accept_mutator(&mutator);
  return s;
}
